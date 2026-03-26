from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict

from models.schemas import DocumentSample, DocumentType, ExtractionFields


DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%d-%m-%Y",
    "%m/%d/%Y",
    "%d %b %Y",
    "%d %B %Y",
]


@dataclass(frozen=True)
class GradingBreakdown:
    classification_accuracy: float
    extraction_accuracy: float
    completeness: float
    final_score: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "classification_accuracy": self.classification_accuracy,
            "extraction_accuracy": self.extraction_accuracy,
            "completeness": self.completeness,
            "final_score": self.final_score,
        }


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", value.strip().lower())


def _parse_amount(value: str | None) -> float | None:
    if not value:
        return None
    cleaned = re.sub(r"[^\d.,-]", "", value)
    if cleaned.count(",") > 0 and cleaned.count(".") > 0:
        if cleaned.rfind(",") > cleaned.rfind("."):
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:
            cleaned = cleaned.replace(",", "")
    else:
        cleaned = cleaned.replace(",", "")
    try:
        return float(cleaned)
    except ValueError:
        return None


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    raw = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def _field_score(pred: ExtractionFields, truth: ExtractionFields) -> float:
    scores = []

    vendor_match = _normalize_text(pred.vendor_name) == _normalize_text(truth.vendor_name)
    scores.append(1.0 if vendor_match else 0.0)

    pred_amount = _parse_amount(pred.total_amount)
    true_amount = _parse_amount(truth.total_amount)
    amount_match = (
        pred_amount is not None
        and true_amount is not None
        and abs(pred_amount - true_amount) < 0.01
    )
    scores.append(1.0 if amount_match else 0.0)

    pred_date = _parse_date(pred.date)
    true_date = _parse_date(truth.date)
    date_match = pred_date is not None and true_date is not None and pred_date.date() == true_date.date()
    scores.append(1.0 if date_match else 0.0)

    return sum(scores) / len(scores)


def _completeness(pred: ExtractionFields) -> float:
    populated = [pred.vendor_name, pred.total_amount, pred.date]
    count = sum(1 for value in populated if bool(value and value.strip()))
    return count / 3.0


def grade_episode(
    sample: DocumentSample,
    predicted_type: DocumentType,
    predicted_fields: ExtractionFields,
) -> GradingBreakdown:
    classification_accuracy = 1.0 if predicted_type == sample.document_type else 0.0
    extraction_accuracy = _field_score(predicted_fields, sample.ground_truth)
    completeness = _completeness(predicted_fields)

    if sample.difficulty == "easy":
        final_score = classification_accuracy
    elif sample.difficulty == "medium":
        final_score = 0.8 * extraction_accuracy + 0.2 * completeness
    else:
        final_score = (
            0.4 * classification_accuracy
            + 0.4 * extraction_accuracy
            + 0.2 * completeness
        )

    return GradingBreakdown(
        classification_accuracy=classification_accuracy,
        extraction_accuracy=extraction_accuracy,
        completeness=completeness,
        final_score=round(final_score, 4),
    )
