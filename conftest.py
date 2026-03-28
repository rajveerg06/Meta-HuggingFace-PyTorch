"""
Shared pytest fixtures for the OpenEnv invoice/receipt test suite.
All tests should import fixtures from here via conftest.py auto-discovery.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

# Ensure the project root is on sys.path regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from env import OpenEnvInvoiceEnv
from models.schemas import DocumentSample, DocumentType, ExtractionFields


# ─────────────────────────────────────────────────────────────────────────────
# Environment fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def env() -> OpenEnvInvoiceEnv:
    """A single environment instance shared across all tests (session scope)."""
    return OpenEnvInvoiceEnv(data_root="data", seed=42)


@pytest.fixture()
def fresh_env() -> OpenEnvInvoiceEnv:
    """A fresh environment for tests that need isolated state."""
    return OpenEnvInvoiceEnv(data_root="data", seed=0)


# ─────────────────────────────────────────────────────────────────────────────
# Sample fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def receipt_sample() -> DocumentSample:
    return DocumentSample(
        sample_id="test-receipt-001",
        source_dataset="test",
        split="test",
        difficulty="hard",
        ocr_text="ACME MARKET\nDate: 2024-01-15\nTotal: $25.50\nThank you",
        document_type=DocumentType.RECEIPT,
        ground_truth=ExtractionFields(
            vendor_name="ACME MARKET",
            total_amount="$25.50",
            date="2024-01-15",
        ),
    )


@pytest.fixture(scope="session")
def invoice_sample() -> DocumentSample:
    return DocumentSample(
        sample_id="test-invoice-001",
        source_dataset="test",
        split="test",
        difficulty="hard",
        ocr_text="TAX INVOICE\nVendor: Contoso Ltd\nDate: 2023-06-30\nTotal Due: EUR 1,200.00",
        document_type=DocumentType.INVOICE,
        ground_truth=ExtractionFields(
            vendor_name="Contoso Ltd",
            total_amount="EUR 1,200.00",
            date="2023-06-30",
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# API client fixture
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def api_client() -> TestClient:
    """FastAPI TestClient — shares the same process, no network I/O needed."""
    from api.server import app

    return TestClient(app)
