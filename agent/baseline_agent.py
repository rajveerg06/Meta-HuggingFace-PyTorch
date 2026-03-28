from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from env import OpenEnvInvoiceEnv
from models.schemas import Action, ActionType, BenchmarkResult, EpisodeRecord, Observation

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Heuristic Agent
# ─────────────────────────────────────────────────────────────────────────────


class BaselineHeuristicAgent:
    """
    Rule-based heuristic agent that processes invoice/receipt OCR text.

    Uses keyword frequency scoring for classification and regex patterns
    for field extraction. No external API required.
    """

    _INVOICE_MARKERS = [
        "invoice", "invoice no", "invoice number", "bill to",
        "tax invoice", "total due", "net 30", "vat", "gst reg",
        "purchase order", "payment terms",
    ]
    _RECEIPT_MARKERS = [
        "receipt", "thank you", "cash", "change", "paid",
        "cashier", "pos#", "gst reg no",
    ]

    def classify_document(self, text: str) -> str:
        text_l = text.lower()
        invoice_score = sum(marker in text_l for marker in self._INVOICE_MARKERS)
        receipt_score = sum(marker in text_l for marker in self._RECEIPT_MARKERS)
        return "invoice" if invoice_score >= receipt_score else "receipt"

    def extract_fields(self, text: str) -> Dict[str, str]:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        return {
            "vendor_name": self._extract_vendor(lines),
            "total_amount": self._extract_amount(text),
            "date": self._extract_date(text),
        }

    def validate_fields(self, fields: Dict[str, str]) -> bool:
        return all(fields.get(k, "").strip() for k in ["vendor_name", "total_amount", "date"])

    # ── private helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_vendor(lines: List[str]) -> str:
        skip_tokens = {
            "invoice", "receipt", "tax invoice", "bill to", "ship to",
            "total", "subtotal", "date", "purchase order", "pay to",
            "dear", "attn", "re:",
        }
        for line in lines[:10]:
            lline = line.lower()
            if any(token in lline for token in skip_tokens):
                continue
            # Skip lines that look like addresses or pure numbers
            if len(line) >= 3 and not re.search(r"\d{3,}", line):
                return line
        return lines[0] if lines else ""

    @staticmethod
    def _extract_amount(text: str) -> str:
        patterns = [
            r"(?:grand\s*total|total\s*(?:payable|due|amount)?)\s*[:\-]?\s*([\w$€£¥₹R\s]*\d[\d,]*\.?\d*)",
            r"(?:total)\s*[:\-]?\s*([\w$€£¥₹R\s]*\d[\d,]*\.?\d*)",
            r"(?:amount\s*due)\s*[:\-]?\s*([\w$€£¥₹R\s]*\d[\d,]*\.?\d*)",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                candidate = matches[-1].strip()
                # Ensure it contains digits
                if re.search(r"\d", candidate):
                    return candidate
        # Fallback: last currency-bearing number in document
        fallback = re.findall(r"[\$RM€£¥₹AUD CAD INR GBP EUR USD AED]?\s?\d[\d,]*\.\d{2}", text)
        return fallback[-1].strip() if fallback else ""

    @staticmethod
    def _extract_date(text: str) -> str:
        patterns = [
            r"(?:date|issued?|invoice\s*date)\s*[:\-]?\s*(\d{4}-\d{2}-\d{2})",
            r"(?:date|issued?|invoice\s*date)\s*[:\-]?\s*(\d{2}/\d{2}/\d{4})",
            r"(?:date|issued?|invoice\s*date)\s*[:\-]?\s*(\d{2}-\d{2}-\d{4})",
            r"\b(\d{4}-\d{2}-\d{2})\b",
            r"\b(\d{2}/\d{2}/\d{4})\b",
            r"\b(\d{2}-\d{2}-\d{4})\b",
        ]
        for pat in patterns:
            m = re.search(pat, text, flags=re.IGNORECASE)
            if m:
                return m.group(1)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark(
    seed: int = 42,
    episodes_per_level: int = 3,
    agent_name: str = "heuristic",
    use_openai: bool = False,
) -> BenchmarkResult:
    """
    Run a full benchmark across all difficulty levels.

    Args:
        seed: Random seed for reproducibility.
        episodes_per_level: Number of episodes per difficulty.
        agent_name: Tag for results ('heuristic' or 'openai').
        use_openai: If True, uses the OpenAI agent (requires OPENAI_API_KEY).

    Returns:
        BenchmarkResult with per-episode records and summary scores.
    """
    random.seed(seed)

    env = OpenEnvInvoiceEnv(data_root="data", seed=seed)

    if use_openai:
        from agent.openai_agent import OpenAIAgent  # noqa: PLC0415

        agent: object = OpenAIAgent()
        agent_name = "openai"
    else:
        agent = BaselineHeuristicAgent()

    episodes: List[EpisodeRecord] = []

    for difficulty in ["easy", "medium", "hard"]:
        for idx in range(episodes_per_level):
            t0 = time.perf_counter()
            state = env.reset(difficulty=difficulty, sample_index=idx)
            obs: Observation = state.observation

            if isinstance(agent, BaselineHeuristicAgent):
                _run_heuristic_episode(env, agent, obs, difficulty)
            else:
                from agent.openai_agent import OpenAIAgent  # noqa: PLC0415

                assert isinstance(agent, OpenAIAgent)
                _run_openai_episode(env, agent, obs, difficulty)

            # Always call finish at the end
            state = env.step(Action(action_type=ActionType.FINISH, payload={}))
            elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

            episodes.append(
                EpisodeRecord(
                    difficulty=difficulty,
                    sample_id=state.observation.sample_id,
                    total_reward=state.reward.total_reward,
                    grader_score=state.reward.grader_score,
                    steps_taken=state.observation.steps_taken,
                    elapsed_ms=elapsed_ms,
                )
            )
            logger.info(
                "[%s] %s idx=%d → grader=%.4f reward=%.4f steps=%d (%.0fms)",
                difficulty,
                state.observation.sample_id,
                idx,
                state.reward.grader_score,
                state.reward.total_reward,
                state.observation.steps_taken,
                elapsed_ms,
            )

    # Aggregate
    by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for ep in episodes:
        by_difficulty[ep.difficulty].append(ep.grader_score)

    average_score: Dict[str, float] = {
        key: round(sum(vals) / max(1, len(vals)), 4)
        for key, vals in by_difficulty.items()
    }
    all_scores = [s for vals in by_difficulty.values() for s in vals]
    average_score["overall"] = round(sum(all_scores) / max(1, len(all_scores)), 4)

    return BenchmarkResult(
        agent=agent_name,
        seed=seed,
        episodes_per_level=episodes_per_level,
        episodes=episodes,
        average_score=average_score,
    )


def _run_heuristic_episode(
    env: OpenEnvInvoiceEnv,
    agent: BaselineHeuristicAgent,
    obs: Observation,
    difficulty: str,
) -> None:
    """Execute one heuristic episode (does NOT call finish)."""
    fields: Dict[str, str] = {}

    if difficulty in {"easy", "hard"}:
        doc_type = agent.classify_document(obs.ocr_text)
        state = env.step(
            Action(
                action_type=ActionType.CLASSIFY_DOCUMENT,
                payload={"document_type": doc_type},
            )
        )
        obs = state.observation

    if difficulty in {"medium", "hard"}:
        fields = agent.extract_fields(obs.ocr_text)
        state = env.step(
            Action(
                action_type=ActionType.EXTRACT_FIELDS,
                payload={"fields": fields},
            )
        )

    if difficulty == "hard":
        env.step(
            Action(
                action_type=ActionType.VALIDATE_FIELDS,
                payload={"is_valid": agent.validate_fields(fields)},
            )
        )


def _run_openai_episode(
    env: OpenEnvInvoiceEnv,
    agent: "OpenAIAgent",  # type: ignore[name-defined]
    obs: Observation,
    difficulty: str,
) -> None:
    """Execute one OpenAI-powered episode (does NOT call finish)."""
    fields: Dict[str, str] = {}

    if difficulty in {"easy", "hard"}:
        doc_type = agent.classify_document(obs.ocr_text)
        state = env.step(
            Action(
                action_type=ActionType.CLASSIFY_DOCUMENT,
                payload={"document_type": doc_type},
            )
        )
        obs = state.observation

    if difficulty in {"medium", "hard"}:
        fields = agent.extract_fields(obs.ocr_text)
        env.step(
            Action(
                action_type=ActionType.EXTRACT_FIELDS,
                payload={"fields": fields},
            )
        )

    if difficulty == "hard":
        env.step(
            Action(
                action_type=ActionType.VALIDATE_FIELDS,
                payload={"is_valid": agent.validate_fields(fields)},
            )
        )
