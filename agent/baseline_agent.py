from __future__ import annotations

import json
import os
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional

from env import OpenEnvInvoiceEnv
from models.schemas import Action, ActionType, Observation


@dataclass
class EpisodeResult:
    difficulty: str
    sample_id: str
    total_reward: float
    grader_score: float


class BaselineHeuristicAgent:
    def __init__(self, use_openai: bool = False) -> None:
        self.use_openai = use_openai and bool(os.getenv("OPENAI_API_KEY"))

    def classify_document(self, text: str) -> str:
        text_l = text.lower()
        invoice_markers = ["invoice", "invoice no", "bill to", "tax invoice"]
        receipt_markers = ["receipt", "thank you", "cash", "change", "gst reg"]

        invoice_score = sum(marker in text_l for marker in invoice_markers)
        receipt_score = sum(marker in text_l for marker in receipt_markers)

        return "invoice" if invoice_score >= receipt_score else "receipt"

    def extract_fields(self, text: str) -> Dict[str, str]:
        lines = [line.strip() for line in text.splitlines() if line.strip()]

        vendor_name = self._extract_vendor(lines)
        total_amount = self._extract_amount(text)
        date = self._extract_date(text)

        return {
            "vendor_name": vendor_name,
            "total_amount": total_amount,
            "date": date,
        }

    @staticmethod
    def _extract_vendor(lines: List[str]) -> str:
        skip = {
            "invoice",
            "receipt",
            "tax invoice",
            "bill to",
            "ship to",
            "total",
            "subtotal",
        }
        for line in lines[:8]:
            lline = line.lower()
            if any(token in lline for token in skip):
                continue
            if len(line) >= 3 and not re.search(r"\d{2,}", line):
                return line
        return lines[0] if lines else ""

    @staticmethod
    def _extract_amount(text: str) -> str:
        patterns = [
            r"(?:total\s*(?:due|amount)?\s*[:\-]?\s*)([\$RM€£]?\s?\d+[\d,]*\.\d{2})",
            r"([\$RM€£]?\s?\d+[\d,]*\.\d{2})\s*$",
        ]
        for pat in patterns:
            matches = re.findall(pat, text, flags=re.IGNORECASE | re.MULTILINE)
            if matches:
                return matches[-1].replace("  ", " ").strip()
        return ""

    @staticmethod
    def _extract_date(text: str) -> str:
        patterns = [
            r"(\d{4}-\d{2}-\d{2})",
            r"(\d{2}/\d{2}/\d{4})",
            r"(\d{2}-\d{2}-\d{4})",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1)
        return ""

    def validate_fields(self, fields: Dict[str, str]) -> bool:
        return all(fields.get(k, "").strip() for k in ["vendor_name", "total_amount", "date"])


def run_benchmark(seed: int = 42, episodes_per_level: int = 3) -> Dict[str, object]:
    random.seed(seed)
    env = OpenEnvInvoiceEnv(data_root="data", seed=seed)
    agent = BaselineHeuristicAgent(use_openai=False)

    results: List[EpisodeResult] = []

    for difficulty in ["easy", "medium", "hard"]:
        for idx in range(episodes_per_level):
            state = env.reset(difficulty=difficulty, sample_index=idx)
            obs: Observation = state.observation

            if difficulty in {"easy", "hard"}:
                doc_type = agent.classify_document(obs.ocr_text)
                state = env.step(
                    Action(
                        action_type=ActionType.CLASSIFY_DOCUMENT,
                        payload={"document_type": doc_type},
                    )
                )
                obs = state.observation

            fields = {}
            if difficulty in {"medium", "hard"}:
                fields = agent.extract_fields(obs.ocr_text)
                state = env.step(
                    Action(
                        action_type=ActionType.EXTRACT_FIELDS,
                        payload={"fields": fields},
                    )
                )
                obs = state.observation

            if difficulty == "hard":
                state = env.step(
                    Action(
                        action_type=ActionType.VALIDATE_FIELDS,
                        payload={"is_valid": agent.validate_fields(fields)},
                    )
                )

            state = env.step(Action(action_type=ActionType.FINISH, payload={}))

            results.append(
                EpisodeResult(
                    difficulty=difficulty,
                    sample_id=state.observation.sample_id,
                    total_reward=state.reward.total_reward,
                    grader_score=state.reward.grader_score,
                )
            )

    by_difficulty: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for item in results:
        by_difficulty[item.difficulty].append(item.grader_score)

    summary = {
        "seed": seed,
        "episodes_per_level": episodes_per_level,
        "results": [item.__dict__ for item in results],
        "average_score": {
            key: round(sum(values) / max(1, len(values)), 4)
            for key, values in by_difficulty.items()
        },
    }
    summary["average_score"]["overall"] = round(
        sum(summary["average_score"][k] for k in ["easy", "medium", "hard"]) / 3.0,
        4,
    )
    return summary


if __name__ == "__main__":
    report = run_benchmark(seed=42, episodes_per_level=3)
    print(json.dumps(report, indent=2))
