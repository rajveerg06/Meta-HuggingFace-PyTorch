"""
Tests for the deterministic grader (graders/scoring.py).
"""
from __future__ import annotations

import pytest

from graders.scoring import (
    GradingBreakdown,
    _amount_score,
    _date_score,
    _vendor_score,
    grade_episode,
)
from models.schemas import DocumentSample, DocumentType, ExtractionFields


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────


def _make_sample(
    difficulty: str = "hard",
    doc_type: str = "invoice",
    vendor: str = "ACME Corp",
    amount: str = "1,200.00",
    date: str = "2024-01-15",
) -> DocumentSample:
    return DocumentSample(
        sample_id="g-test-001",
        source_dataset="test",
        split="test",
        difficulty=difficulty,  # type: ignore[arg-type]
        ocr_text="",
        document_type=DocumentType(doc_type),
        ground_truth=ExtractionFields(
            vendor_name=vendor,
            total_amount=amount,
            date=date,
        ),
    )


def _make_pred(
    vendor: str | None = None,
    amount: str | None = None,
    date: str | None = None,
) -> ExtractionFields:
    return ExtractionFields(vendor_name=vendor, total_amount=amount, date=date)


# ─────────────────────────────────────────────────────────────────────────────
# Individual field scoring
# ─────────────────────────────────────────────────────────────────────────────


class TestVendorScore:
    def test_exact_match(self) -> None:
        assert _vendor_score("ACME Corp", "ACME Corp") == 1.0

    def test_case_insensitive(self) -> None:
        assert _vendor_score("acme corp", "ACME CORP") == 1.0

    def test_empty_pred_returns_zero(self) -> None:
        assert _vendor_score(None, "ACME Corp") == 0.0

    def test_empty_truth_returns_zero(self) -> None:
        assert _vendor_score("ACME Corp", None) == 0.0

    def test_completely_wrong(self) -> None:
        assert _vendor_score("XYZ Random", "ACME Corp") == 0.0


class TestAmountScore:
    def test_exact_match(self) -> None:
        assert _amount_score("1200.00", "1200.00") == 1.0

    def test_currency_agnostic(self) -> None:
        assert _amount_score("$1,200.00", "1200.00") == 1.0

    def test_european_format(self) -> None:
        # 1.200,00 (European) == 1200.00
        assert _amount_score("1.200,00", "1200.00") == 1.0

    def test_rm_currency(self) -> None:
        assert _amount_score("RM 45.90", "45.90") == 1.0

    def test_eur_currency(self) -> None:
        assert _amount_score("EUR 2,190.40", "2190.40") == 1.0

    def test_within_tolerance(self) -> None:
        # Less than 1 cent difference
        assert _amount_score("25.499", "25.50") == 1.0

    def test_wrong_amount(self) -> None:
        assert _amount_score("999.00", "1200.00") == 0.0

    def test_empty_pred(self) -> None:
        assert _amount_score(None, "100.00") == 0.0

    def test_inr_large_amount(self) -> None:
        assert _amount_score("INR 578,200.00", "578200.00") == 1.0


class TestDateScore:
    def test_iso_format(self) -> None:
        assert _date_score("2024-01-15", "2024-01-15") == 1.0

    def test_slash_format(self) -> None:
        assert _date_score("15/01/2024", "2024-01-15") == 1.0

    def test_dash_format(self) -> None:
        assert _date_score("15-01-2024", "2024-01-15") == 1.0

    def test_wrong_date(self) -> None:
        assert _date_score("2023-06-10", "2024-01-15") == 0.0

    def test_empty_date(self) -> None:
        assert _date_score(None, "2024-01-15") == 0.0

    def test_unparseable_format(self) -> None:
        assert _date_score("January 15, 2024", "2024-01-15") == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# grade_episode
# ─────────────────────────────────────────────────────────────────────────────


class TestGradeEpisode:
    def test_perfect_easy_score(self) -> None:
        sample = _make_sample(difficulty="easy")
        result = grade_episode(sample, DocumentType.INVOICE, _make_pred())
        assert result.classification_accuracy == 1.0
        assert result.final_score == 1.0

    def test_wrong_classification_easy(self) -> None:
        sample = _make_sample(difficulty="easy", doc_type="invoice")
        result = grade_episode(sample, DocumentType.RECEIPT, _make_pred())
        assert result.classification_accuracy == 0.0
        assert result.final_score == 0.0

    def test_perfect_medium_score(self) -> None:
        sample = _make_sample(difficulty="medium", vendor="ACME Corp", amount="1,200.00", date="2024-01-15")
        pred = _make_pred(vendor="ACME Corp", amount="1200.00", date="2024-01-15")
        result = grade_episode(sample, DocumentType.INVOICE, pred)
        assert result.extraction_accuracy == 1.0
        assert result.completeness == 1.0
        assert result.final_score == pytest.approx(1.0)

    def test_partial_medium_score(self) -> None:
        sample = _make_sample(difficulty="medium", vendor="ACME Corp", amount="1,200.00", date="2024-01-15")
        # Only vendor_name correct
        pred = _make_pred(vendor="ACME Corp", amount="WRONG", date="WRONG")
        result = grade_episode(sample, DocumentType.INVOICE, pred)
        assert 0.0 < result.extraction_accuracy < 1.0

    def test_hard_full_correct(self) -> None:
        sample = _make_sample(difficulty="hard", doc_type="receipt", vendor="JJ MART", amount="6.00", date="18/02/2024")
        pred = _make_pred(vendor="JJ MART", amount="RM 6.00", date="18/02/2024")
        result = grade_episode(sample, DocumentType.RECEIPT, pred)
        assert result.classification_accuracy == 1.0
        assert result.final_score == pytest.approx(
            0.4 * 1.0 + 0.4 * result.extraction_accuracy + 0.2 * result.completeness,
            abs=0.001,
        )

    def test_hard_wrong_classification(self) -> None:
        sample = _make_sample(difficulty="hard", doc_type="invoice")
        pred = _make_pred(vendor="Vendor A", amount="100.00", date="2024-01-01")
        result = grade_episode(sample, DocumentType.RECEIPT, pred)  # Wrong!
        assert result.classification_accuracy == 0.0
        assert result.final_score < 1.0

    def test_result_is_frozen(self, receipt_sample: DocumentSample) -> None:
        result = grade_episode(receipt_sample, DocumentType.RECEIPT, _make_pred())
        with pytest.raises(AttributeError):
            result.final_score = 999.0  # type: ignore[misc]

    def test_grade_returns_grading_breakdown(self, receipt_sample: DocumentSample) -> None:
        result = grade_episode(receipt_sample, DocumentType.RECEIPT, _make_pred())
        assert isinstance(result, GradingBreakdown)

    def test_as_dict_has_required_keys(self, receipt_sample: DocumentSample) -> None:
        result = grade_episode(receipt_sample, DocumentType.RECEIPT, _make_pred())
        d = result.as_dict()
        required = {
            "classification_accuracy", "extraction_accuracy", "completeness",
            "final_score", "vendor_name_score", "total_amount_score", "date_score",
        }
        assert required.issubset(d.keys())

    def test_score_range_always_valid(self) -> None:
        """Final score should always be in [0.0, 1.0]."""
        for difficulty in ["easy", "medium", "hard"]:
            sample = _make_sample(difficulty=difficulty)
            result = grade_episode(sample, DocumentType.UNKNOWN, _make_pred())
            assert 0.0 <= result.final_score <= 1.0

    def test_completeness_all_none(self) -> None:
        sample = _make_sample(difficulty="medium")
        result = grade_episode(sample, DocumentType.INVOICE, _make_pred())
        assert result.completeness == 0.0

    def test_completeness_partial(self) -> None:
        sample = _make_sample(difficulty="medium")
        pred = _make_pred(vendor="X", amount=None, date=None)
        result = grade_episode(sample, DocumentType.INVOICE, pred)
        assert result.completeness == pytest.approx(1 / 3)
