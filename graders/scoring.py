from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Optional

from models.schemas import DocumentSample, DocumentType, ExtractionFields

# Try to import rapidfuzz for fuzzy vendor matching.
# Falls back to exact matching if not installed.
try:
    from rapidfuzz import fuzz as _fuzz  # type: ignore

    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False

# Vendor name similarity threshold (0–100 scale used by rapidfuzz)
_VENDOR_FUZZY_THRESHOLD = 85.0

DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y/%m/%d",
]


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclass
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class GradingBreakdown:
    classification_accuracy: float
    extraction_accuracy: float
    completeness: float
    final_score: float
    # Per-field detail
    vendor_name_score: float = 0.0
    total_amount_score: float = 0.0
    date_score: float = 0.0
    fuzzy_vendor: bool = False

    def as_dict(self) -> Dict[str, float]:
        return {
            "classification_accuracy": self.classification_accuracy,
            "extraction_accuracy": self.extraction_accuracy,
            "completeness": self.completeness,
            "final_score": self.final_score,
            "vendor_name_score": self.vendor_name_score,
            "total_amount_score": self.total_amount_score,
            "date_score": self.date_score,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Normalisation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_amount(value: str | None) -> Optional[float]:
    if not value:
        return None
    # Strip currency symbols and normalise separators
    cleaned = re.sub(r"[^\d.,-]", "", value)
    if not cleaned:
        return None
    if cleaned.count(",") > 0 and cleaned.count(".") > 0:
        # Determine decimal separator by position of last occurrence
        if cleaned.rfind(",") > cleaned.rfind("."):
            # European format: 1.234,56 → 1234.56
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            # US format: 1,234.56 → 1234.56
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(value: str | None) -> Optional[datetime]:
    if not value:
        return None
    raw = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-field scoring
# ─────────────────────────────────────────────────────────────────────────────


def _vendor_score(pred: Optional[str], truth: Optional[str]) -> float:
    """
    Score vendor name match.

    If rapidfuzz is available: full credit at ≥85 similarity ratio,
    partial credit (0.5) at ≥70, otherwise 0.
    Falls back to exact normalised match when rapidfuzz is not installed.
    """
    pred_norm = _normalize_text(pred)
    truth_norm = _normalize_text(truth)

    if not pred_norm or not truth_norm:
        return 0.0

    if pred_norm == truth_norm:
        return 1.0

    if _FUZZY_AVAILABLE:
        ratio = _fuzz.ratio(pred_norm, truth_norm)
        if ratio >= _VENDOR_FUZZY_THRESHOLD:
            return 1.0
        if ratio >= 70.0:
            return 0.5
        return 0.0

    return 0.0  # Exact match already checked above; no fuzzy fallback


def _amount_score(pred: Optional[str], truth: Optional[str]) -> float:
    pred_val = _parse_amount(pred)
    truth_val = _parse_amount(truth)
    if pred_val is None or truth_val is None:
        return 0.0
    # Allow ±1 cent tolerance for float comparison
    return 1.0 if abs(pred_val - truth_val) < 0.01 else 0.0


def _date_score(pred: Optional[str], truth: Optional[str]) -> float:
    pred_dt = _parse_date(pred)
    truth_dt = _parse_date(truth)
    if pred_dt is None or truth_dt is None:
        return 0.0
    return 1.0 if pred_dt.date() == truth_dt.date() else 0.0


def _field_score(pred: ExtractionFields, truth: ExtractionFields) -> tuple[float, float, float]:
    """Return per-field scores: (vendor_score, amount_score, date_score)."""
    v = _vendor_score(pred.vendor_name, truth.vendor_name)
    a = _amount_score(pred.total_amount, truth.total_amount)
    d = _date_score(pred.date, truth.date)
    return v, a, d


def _completeness(pred: ExtractionFields) -> float:
    populated = [pred.vendor_name, pred.total_amount, pred.date]
    count = sum(1 for value in populated if bool(value and value.strip()))
    return count / 3.0


# ─────────────────────────────────────────────────────────────────────────────
# Main grading function
# ─────────────────────────────────────────────────────────────────────────────


def grade_episode(
    sample: DocumentSample,
    predicted_type: DocumentType,
    predicted_fields: ExtractionFields,
) -> GradingBreakdown:
    """
    Deterministically grade one episode against ground truth.

    Scoring weights by difficulty:
        easy:   classification only
        medium: 0.8 × extraction + 0.2 × completeness
        hard:   0.4 × classification + 0.4 × extraction + 0.2 × completeness

    Field extraction accuracy is the mean of per-field scores (vendor, amount, date).
    Vendor matching uses fuzzy similarity when rapidfuzz is installed.

    Returns:
        GradingBreakdown with all sub-scores and final_score ∈ [0.0, 1.0].
    """
    classification_accuracy = 1.0 if predicted_type == sample.document_type else 0.0

    v_score, a_score, d_score = _field_score(predicted_fields, sample.ground_truth)
    extraction_accuracy = (v_score + a_score + d_score) / 3.0
    completeness = _completeness(predicted_fields)

    if sample.difficulty == "easy":
        final_score = classification_accuracy
    elif sample.difficulty == "medium":
        final_score = 0.8 * extraction_accuracy + 0.2 * completeness
    else:  # hard
        final_score = (
            0.4 * classification_accuracy
            + 0.4 * extraction_accuracy
            + 0.2 * completeness
        )

    return GradingBreakdown(
        classification_accuracy=round(classification_accuracy, 4),
        extraction_accuracy=round(extraction_accuracy, 4),
        completeness=round(completeness, 4),
        final_score=round(final_score, 4),
        vendor_name_score=round(v_score, 4),
        total_amount_score=round(a_score, 4),
        date_score=round(d_score, 4),
        fuzzy_vendor=_FUZZY_AVAILABLE,
    )
