from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from graders import grade_episode
from models.schemas import (
    Action,
    ActionType,
    DocumentSample,
    DocumentType,
    EpisodeState,
    ExtractionFields,
    Observation,
    ResetOptions,
    Reward,
)
from tasks import TASK_DEFINITIONS

from .dataset_loader import DatasetLoader
from .reward import step_reward_for_action


class OpenEnvInvoiceEnv:
    def __init__(self, data_root: str | Path = "data", seed: int = 42, max_steps: int = 8) -> None:
        self.data_root = Path(data_root)
        self.seed = seed
        self.max_steps = max_steps

        self.dataset = DatasetLoader(self.data_root).load()
        self.samples_by_difficulty: Dict[str, List[DocumentSample]] = defaultdict(list)
        for sample in self.dataset:
            self.samples_by_difficulty[sample.difficulty].append(sample)

        self._cursors: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        self._sample: Optional[DocumentSample] = None
        self._difficulty = "easy"
        self._predicted_type: DocumentType = DocumentType.UNKNOWN
        self._extracted_fields = ExtractionFields()
        self._validation_passed = False
        self._done = False
        self._steps_taken = 0
        self._total_reward = 0.0
        self._last_action: Optional[ActionType] = None
        self._last_reward = Reward()
        self._action_counts: Dict[ActionType, int] = defaultdict(int)

    def reset(
        self,
        options: Optional[ResetOptions] = None,
        difficulty: Optional[str] = None,
        sample_index: Optional[int] = None,
    ) -> EpisodeState:
        if options is not None:
            difficulty = options.difficulty
            sample_index = options.sample_index

        self._difficulty = difficulty or "easy"
        if self._difficulty not in TASK_DEFINITIONS:
            raise ValueError(f"Unsupported difficulty '{self._difficulty}'.")

        pool = self.samples_by_difficulty[self._difficulty]
        if not pool:
            raise ValueError(f"No samples available for difficulty {self._difficulty}.")

        if sample_index is None:
            idx = self._cursors[self._difficulty] % len(pool)
            self._cursors[self._difficulty] += 1
        else:
            idx = sample_index % len(pool)
        self._sample = pool[idx]

        self._predicted_type = DocumentType.UNKNOWN
        self._extracted_fields = ExtractionFields()
        self._validation_passed = False
        self._done = False
        self._steps_taken = 0
        self._total_reward = 0.0
        self._last_action = None
        self._last_reward = Reward()
        self._action_counts = defaultdict(int)

        return EpisodeState(observation=self._observation(), reward=self._last_reward)

    def step(self, action: Action) -> EpisodeState:
        if self._sample is None:
            raise RuntimeError("Environment has not been reset.")
        if self._done:
            return EpisodeState(observation=self._observation(), reward=self._last_reward)

        self._steps_taken += 1
        self._last_action = action.action_type
        self._action_counts[action.action_type] += 1

        loop_penalty = max(0.0, (self._action_counts[action.action_type] - 1) * 0.05)
        action_correct = False

        if action.action_type == ActionType.CLASSIFY_DOCUMENT:
            action_correct = self._handle_classification(action)
        elif action.action_type == ActionType.EXTRACT_FIELDS:
            action_correct = self._handle_extraction(action)
        elif action.action_type == ActionType.VALIDATE_FIELDS:
            action_correct = self._handle_validation(action)
        elif action.action_type == ActionType.FINISH:
            self._done = True
            action_correct = True
        else:
            action_correct = False

        step_reward = step_reward_for_action(
            action=action.action_type,
            action_correct=action_correct,
            loop_penalty=loop_penalty,
            sample=self._sample,
        )

        grader = grade_episode(self._sample, self._predicted_type, self._extracted_fields)

        if action.action_type == ActionType.FINISH:
            # Final reward emphasizes measurable task quality.
            step_reward += 0.6 * grader.final_score

        if self._steps_taken >= self.max_steps:
            self._done = True
            step_reward -= 0.1

        self._total_reward = round(self._total_reward + step_reward, 4)
        self._last_reward = Reward(
            step_reward=round(step_reward, 4),
            total_reward=self._total_reward,
            grader_score=grader.final_score,
            details=grader.as_dict(),
        )

        return EpisodeState(observation=self._observation(), reward=self._last_reward)

    def state(self) -> EpisodeState:
        if self._sample is None:
            raise RuntimeError("Environment has not been reset.")
        return EpisodeState(observation=self._observation(), reward=self._last_reward)

    def _handle_classification(self, action: Action) -> bool:
        predicted = str(action.payload.get("document_type", "unknown")).lower().strip()
        if predicted not in {"invoice", "receipt"}:
            self._predicted_type = DocumentType.UNKNOWN
            return False
        self._predicted_type = DocumentType(predicted)
        return self._predicted_type == self._sample.document_type

    def _handle_extraction(self, action: Action) -> bool:
        raw_fields = action.payload.get("fields", {})
        if not isinstance(raw_fields, dict):
            return False

        fields = ExtractionFields(
            vendor_name=str(raw_fields.get("vendor_name", "")).strip() or None,
            total_amount=str(raw_fields.get("total_amount", "")).strip() or None,
            date=str(raw_fields.get("date", "")).strip() or None,
        )
        self._extracted_fields = fields

        gt = self._sample.ground_truth
        matches = 0
        if self._norm(fields.vendor_name) == self._norm(gt.vendor_name):
            matches += 1
        if self._norm_amount(fields.total_amount) == self._norm_amount(gt.total_amount):
            matches += 1
        if self._norm_date(fields.date) == self._norm_date(gt.date):
            matches += 1
        return matches >= 2

    def _handle_validation(self, action: Action) -> bool:
        vendor_ok = bool(self._extracted_fields.vendor_name and len(self._extracted_fields.vendor_name) >= 2)
        amount_ok = self._norm_amount(self._extracted_fields.total_amount) is not None
        date_ok = self._norm_date(self._extracted_fields.date) is not None

        env_validation = vendor_ok and amount_ok and date_ok
        claimed_valid = bool(action.payload.get("is_valid", False))

        self._validation_passed = env_validation
        return env_validation == claimed_valid

    def _observation(self) -> Observation:
        assert self._sample is not None
        required = TASK_DEFINITIONS[self._difficulty].required_actions
        progress = min(1.0, self._steps_taken / float(max(1, len(required) + 1)))

        return Observation(
            sample_id=self._sample.sample_id,
            difficulty=self._sample.difficulty,
            ocr_text=self._sample.ocr_text,
            predicted_document_type=self._predicted_type,
            extracted_fields=self._extracted_fields,
            validation_passed=self._validation_passed,
            progress=round(progress, 4),
            steps_taken=self._steps_taken,
            max_steps=self.max_steps,
            available_actions=self._available_actions(),
            last_action=self._last_action,
            done=self._done,
            info={
                "task": TASK_DEFINITIONS[self._difficulty].objective,
                "required_actions": [a.value for a in required],
            },
        )

    def _available_actions(self) -> List[ActionType]:
        if self._done:
            return []
        if self._difficulty == "easy":
            return [ActionType.CLASSIFY_DOCUMENT, ActionType.FINISH]
        if self._difficulty == "medium":
            return [ActionType.EXTRACT_FIELDS, ActionType.FINISH]
        return [
            ActionType.CLASSIFY_DOCUMENT,
            ActionType.EXTRACT_FIELDS,
            ActionType.VALIDATE_FIELDS,
            ActionType.FINISH,
        ]

    @staticmethod
    def _norm(value: str | None) -> str:
        if not value:
            return ""
        return re.sub(r"\s+", " ", value.strip().lower())

    @staticmethod
    def _norm_amount(value: str | None) -> str | None:
        if not value:
            return None
        digits = re.sub(r"[^\d.,-]", "", value)
        if not digits:
            return None
        if digits.count(",") > 0 and digits.count(".") > 0:
            if digits.rfind(",") > digits.rfind("."):
                digits = digits.replace(".", "").replace(",", ".")
            else:
                digits = digits.replace(",", "")
        else:
            digits = digits.replace(",", "")
        try:
            return f"{float(digits):.2f}"
        except ValueError:
            return None

    @staticmethod
    def _norm_date(value: str | None) -> str | None:
        if not value:
            return None
        raw = value.strip()
        patterns = [
            r"^(\d{4})-(\d{2})-(\d{2})$",
            r"^(\d{2})/(\d{2})/(\d{4})$",
            r"^(\d{2})-(\d{2})-(\d{4})$",
        ]
        for pat in patterns:
            m = re.match(pat, raw)
            if not m:
                continue
            groups = list(m.groups())
            if pat == patterns[0]:
                yyyy, mm, dd = groups
            else:
                dd, mm, yyyy = groups
            return f"{yyyy}-{mm}-{dd}"
        return None
