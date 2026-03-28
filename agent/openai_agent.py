from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict

logger = logging.getLogger(__name__)

try:
    from openai import OpenAI  # type: ignore

    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False

# Lazy import fallback
_HEURISTIC_FALLBACK_IMPORTED = False


class OpenAIAgent:
    """
    LLM-powered agent using the OpenAI Chat Completions API (GPT-4o-mini).

    Requires OPENAI_API_KEY environment variable.
    Falls back to BaselineHeuristicAgent on any API error to ensure
    benchmark continuity.

    The agent uses JSON-mode structured output to guarantee parseable responses.
    """

    _CLASSIFY_SYSTEM = (
        "You are a document classification expert. "
        "Given OCR text from a financial document, determine if it is an 'invoice' or 'receipt'. "
        "Respond ONLY with valid JSON: {\"document_type\": \"invoice\"} or {\"document_type\": \"receipt\"}."
    )

    _EXTRACT_SYSTEM = (
        "You are a financial document information extractor. "
        "Extract structured fields from the OCR text. "
        "Respond ONLY with valid JSON:\n"
        '{"vendor_name": "<string>", "total_amount": "<string>", "date": "<string>", '
        '"confidence": {"vendor_name": <0.0-1.0>, "total_amount": <0.0-1.0>, "date": <0.0-1.0>}}\n'
        "Use null for fields you cannot find. Preserve the original currency symbol and format for total_amount. "
        "For date, use the format as it appears in the document."
    )

    def __init__(
        self,
        model: str | None = None,
        temperature: float = 0.0,
    ) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "openai package not installed. Run: pip install openai"
            )
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Either set it or use BaselineHeuristicAgent instead."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self._temperature = temperature
        logger.info("OpenAIAgent initialised with model=%s", self._model)

    def classify_document(self, text: str) -> str:
        """Classify OCR text as 'invoice' or 'receipt' using the LLM."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._CLASSIFY_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Classify this document:\n\n{text[:3000]}",
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            result = json.loads(content)
            doc_type = str(result.get("document_type", "receipt")).lower().strip()
            if doc_type not in {"invoice", "receipt"}:
                doc_type = "receipt"
            logger.debug("LLM classified as: %s", doc_type)
            return doc_type
        except Exception as exc:
            logger.warning("OpenAI classify failed (%s), falling back to heuristic.", exc)
            return self._heuristic_fallback().classify_document(text)

    def extract_fields(self, text: str) -> Dict[str, Any]:
        """Extract structured fields from OCR text using the LLM."""
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": self._EXTRACT_SYSTEM},
                    {
                        "role": "user",
                        "content": f"Extract fields from this document:\n\n{text[:3000]}",
                    },
                ],
            )
            content = response.choices[0].message.content or "{}"
            result: Dict[str, Any] = json.loads(content)
            # Sanitise
            fields: Dict[str, Any] = {
                "vendor_name": result.get("vendor_name") or "",
                "total_amount": result.get("total_amount") or "",
                "date": result.get("date") or "",
            }
            if isinstance(result.get("confidence"), dict):
                fields["confidence"] = result["confidence"]
            logger.debug("LLM extracted fields: %s", fields)
            return fields
        except Exception as exc:
            logger.warning("OpenAI extract failed (%s), falling back to heuristic.", exc)
            return self._heuristic_fallback().extract_fields(text)

    def validate_fields(self, fields: Dict[str, Any]) -> bool:
        """Validate that all required fields are non-empty."""
        return all(
            str(fields.get(k, "")).strip()
            for k in ["vendor_name", "total_amount", "date"]
        )

    def _heuristic_fallback(self) -> "BaselineHeuristicAgent":
        from agent.baseline_agent import BaselineHeuristicAgent  # noqa: PLC0415

        return BaselineHeuristicAgent()
