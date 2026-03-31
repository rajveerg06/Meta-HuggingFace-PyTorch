from __future__ import annotations

import json
import os
import random
import time
from statistics import mean
from typing import Any, Dict, List

from openai import OpenAI

from env import OpenEnvInvoiceEnv
from models.schemas import Action, ActionType, EpisodeRecord


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


def _safe_json(content: str) -> Dict[str, Any]:
    try:
        payload = json.loads(content)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass
    return {}


class OpenAIInferenceAgent:
    def __init__(self, client: OpenAI, model: str) -> None:
        self.client = client
        self.model = model

    def classify_document(self, ocr_text: str) -> str:
        prompt = (
            "Classify the financial document OCR text as invoice or receipt. "
            'Return JSON only: {"document_type": "invoice"} or {"document_type": "receipt"}.'
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ocr_text[:3500]},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = _safe_json(content)
        doc_type = str(parsed.get("document_type", "receipt")).strip().lower()
        return doc_type if doc_type in {"invoice", "receipt"} else "receipt"

    def extract_fields(self, ocr_text: str) -> Dict[str, str]:
        prompt = (
            "Extract vendor_name, total_amount, and date from OCR text. "
            "Return JSON only with keys: vendor_name, total_amount, date."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            top_p=1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": ocr_text[:3500]},
            ],
        )
        content = response.choices[0].message.content or "{}"
        parsed = _safe_json(content)
        return {
            "vendor_name": str(parsed.get("vendor_name", "")).strip(),
            "total_amount": str(parsed.get("total_amount", "")).strip(),
            "date": str(parsed.get("date", "")).strip(),
        }

    @staticmethod
    def validate_fields(fields: Dict[str, str]) -> bool:
        return all(fields.get(k, "").strip() for k in ["vendor_name", "total_amount", "date"])


def run_inference(seed: int, episodes_per_level: int) -> Dict[str, Any]:
    api_base_url = _require_env("API_BASE_URL")
    model_name = _require_env("MODEL_NAME")
    hf_token = _require_env("HF_TOKEN")

    random.seed(seed)

    client = OpenAI(base_url=api_base_url, api_key=hf_token)
    agent = OpenAIInferenceAgent(client=client, model=model_name)
    env = OpenEnvInvoiceEnv(data_root="data", seed=seed)

    episodes: List[EpisodeRecord] = []

    for difficulty in ["easy", "medium", "hard"]:
        for idx in range(episodes_per_level):
            t0 = time.perf_counter()
            state = env.reset(difficulty=difficulty, sample_index=idx)
            obs = state.observation
            fields: Dict[str, str] = {}

            if difficulty in {"easy", "hard"}:
                doc_type = agent.classify_document(obs.ocr_text)
                state = env.step_state(
                    Action(action_type=ActionType.CLASSIFY_DOCUMENT, payload={"document_type": doc_type})
                )
                obs = state.observation

            if difficulty in {"medium", "hard"}:
                fields = agent.extract_fields(obs.ocr_text)
                state = env.step_state(
                    Action(action_type=ActionType.EXTRACT_FIELDS, payload={"fields": fields})
                )

            if difficulty == "hard":
                state = env.step_state(
                    Action(action_type=ActionType.VALIDATE_FIELDS, payload={"is_valid": agent.validate_fields(fields)})
                )

            state = env.step_state(Action(action_type=ActionType.FINISH, payload={}))
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

    bucket: Dict[str, List[float]] = {"easy": [], "medium": [], "hard": []}
    for ep in episodes:
        bucket[ep.difficulty].append(ep.grader_score)

    average_score = {
        "easy": round(mean(bucket["easy"]) if bucket["easy"] else 0.0, 4),
        "medium": round(mean(bucket["medium"]) if bucket["medium"] else 0.0, 4),
        "hard": round(mean(bucket["hard"]) if bucket["hard"] else 0.0, 4),
    }
    all_scores = bucket["easy"] + bucket["medium"] + bucket["hard"]
    average_score["overall"] = round(mean(all_scores) if all_scores else 0.0, 4)

    return {
        "agent": "openai",
        "seed": seed,
        "episodes_per_level": episodes_per_level,
        "average_score": average_score,
        "episodes": [ep.model_dump() for ep in episodes],
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run OpenEnv baseline inference over all tasks")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=3, help="Episodes per difficulty level")
    parser.add_argument("--output", type=str, default="inference_results.json")
    args = parser.parse_args()

    result = run_inference(seed=args.seed, episodes_per_level=args.episodes)
    with open(args.output, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps(result["average_score"], indent=2))


if __name__ == "__main__":
    main()
