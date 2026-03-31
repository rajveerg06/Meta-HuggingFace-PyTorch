from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """
    OpenEnv-compatible environment for invoice and receipt document processing.

    Implements the OpenEnv API contract:
        reset()  → EpisodeState
        step()   → EpisodeState
        state()  → EpisodeState

    Additionally exposes Gymnasium-compatible metadata properties:
        action_space    → dict describing the discrete action space
        observation_space → dict describing observation fields
        render()  → str  human-readable state summary
    """

    # Gymnasium-style flat descriptors (no dependency on gymnasium package)
    ACTION_SPACE: Dict[str, Any] = {
        "type": "Discrete",
        "n": len(ActionType),
        "actions": [a.value for a in ActionType],
    }

    OBSERVATION_SPACE: Dict[str, Any] = {
        "sample_id": "str",
        "difficulty": "Literal['easy', 'medium', 'hard']",
        "ocr_text": "str",
        "predicted_document_type": "Literal['invoice', 'receipt', 'unknown']",
        "extracted_fields": {
            "vendor_name": "Optional[str]",
            "total_amount": "Optional[str]",
            "date": "Optional[str]",
        },
        "validation_passed": "bool",
        "progress": "float ∈ [0, 1]",
        "steps_taken": "int",
        "max_steps": "int",
        "available_actions": "List[str]",
        "last_action": "Optional[str]",
        "done": "bool",
        "info": "dict",
    }

    def __init__(
        self,
        data_root: str | Path = "data",
        seed: int = 42,
        max_steps: int = 8,
        split: Optional[str] = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.seed = seed
        self.max_steps = max_steps

        self.dataset = DatasetLoader(self.data_root).load(split=split)
        self.samples_by_difficulty: Dict[str, List[DocumentSample]] = defaultdict(list)
        for sample in self.dataset:
            self.samples_by_difficulty[sample.difficulty].append(sample)

        self._cursors: Dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
        self._history: List[Dict[str, Any]] = []  # Episode action history

        # Internal episode state — initialised in reset()
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

    # ─────────────────────────────────────────────────────────────────────────
    # Gymnasium-compatible properties
    # ─────────────────────────────────────────────────────────────────────────

    @property
    def action_space(self) -> Dict[str, Any]:
        """Returns metadata about the discrete action space."""
        return self.ACTION_SPACE

    @property
    def observation_space(self) -> Dict[str, Any]:
        """Returns metadata about the observation space."""
        return self.OBSERVATION_SPACE

    @property
    def done(self) -> bool:
        """True if the current episode has ended."""
        return self._done

    # ─────────────────────────────────────────────────────────────────────────
    # OpenEnv API
    # ─────────────────────────────────────────────────────────────────────────

    def reset(
        self,
        options: Optional[ResetOptions] = None,
        difficulty: Optional[str] = None,
        sample_index: Optional[int] = None,
    ) -> EpisodeState:
        """
        Reset the environment for a new episode.

        Args:
            options: Structured ResetOptions (difficulty + sample_index + split).
            difficulty: Shorthand override; overridden by ``options`` if provided.
            sample_index: Pin to a specific sample (wraps around pool size).

        Returns:
            EpisodeState with the initial observation.
        """
        if options is not None:
            difficulty = options.difficulty
            sample_index = options.sample_index

        self._difficulty = difficulty or "easy"
        if self._difficulty not in TASK_DEFINITIONS:
            raise ValueError(f"Unsupported difficulty {self._difficulty!r}. Choose: easy, medium, hard.")

        pool = self.samples_by_difficulty[self._difficulty]
        if not pool:
            raise ValueError(
                f"No samples available for difficulty={self._difficulty!r}. "
                "Check your data directory or run scripts/generate_dataset.py."
            )

        if sample_index is None:
            idx = self._cursors[self._difficulty] % len(pool)
            self._cursors[self._difficulty] += 1
        else:
            idx = sample_index % len(pool)
        self._sample = pool[idx]

        # Reset episode state
        self._predicted_type = DocumentType.UNKNOWN
        self._extracted_fields = ExtractionFields()
        self._validation_passed = False
        self._done = False
        self._steps_taken = 0
        self._total_reward = 0.0
        self._last_action = None
        self._last_reward = Reward()
        self._action_counts = defaultdict(int)
        self._history = []

        return EpisodeState(observation=self._observation(), reward=self._last_reward)

    def step(self, action: Action) -> tuple[Observation, Reward, bool, Dict[str, Any]]:
        """
        Execute one action in the current episode.

        Args:
            action: Action to execute (action_type + payload).

        Returns:
            (observation, reward, done, info)
        """
        if self._sample is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")
        if self._done:
            obs = self._observation()
            return obs, self._last_reward, self._done, dict(obs.info)

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
            # Final reward heavily weights actual task quality.
            step_reward += 0.6 * grader.final_score

        if self._steps_taken >= self.max_steps and not self._done:
            self._done = True
            step_reward -= 0.1  # Time-limit truncation penalty

        self._total_reward = round(self._total_reward + step_reward, 4)
        self._last_reward = Reward(
            step_reward=round(step_reward, 4),
            total_reward=self._total_reward,
            grader_score=grader.final_score,
            details=grader.as_dict(),
        )

        # Record to episode history for replay/auditing
        self._history.append(
            {
                "step": self._steps_taken,
                "action": action.action_type.value,
                "payload": action.payload,
                "correct": action_correct,
                "step_reward": round(step_reward, 4),
                "grader_score": grader.final_score,
            }
        )

        obs = self._observation()
        return obs, self._last_reward, self._done, dict(obs.info)

    def step_state(self, action: Action) -> EpisodeState:
        """Compatibility wrapper returning EpisodeState for existing callers."""
        observation, reward, _done, _info = self.step(action)
        return EpisodeState(observation=observation, reward=reward)

    def state(self) -> EpisodeState:
        """Return current episode state without advancing the environment."""
        if self._sample is None:
            raise RuntimeError("Environment has not been reset. Call reset() first.")
        return EpisodeState(observation=self._observation(), reward=self._last_reward)

    def render(self) -> str:
        """Return a human-readable text summary of the current episode state."""
        if self._sample is None:
            return "Environment not reset."

        grader = grade_episode(self._sample, self._predicted_type, self._extracted_fields)
        lines = [
            "═" * 60,
            f"  OpenEnv Invoice/Receipt — Difficulty: {self._difficulty.upper()}",
            "═" * 60,
            f"  Sample     : {self._sample.sample_id}",
            f"  Doc Type   : {self._sample.document_type.value} (GT)",
            f"  Predicted  : {self._predicted_type.value}",
            "  — Extracted Fields —",
            f"    vendor   : {self._extracted_fields.vendor_name or '<none>'}",
            f"    amount   : {self._extracted_fields.total_amount or '<none>'}",
            f"    date     : {self._extracted_fields.date or '<none>'}",
            f"  Validation : {'PASSED' if self._validation_passed else 'NOT YET'}",
            f"  Steps      : {self._steps_taken}/{self.max_steps}",
            f"  Progress   : {self._steps_taken / self.max_steps:.0%}",
            f"  Total Rwrd : {self._total_reward:.4f}",
            f"  Grader     : {grader.final_score:.4f}",
            f"  Done       : {self._done}",
            "═" * 60,
        ]
        return "\n".join(lines)

    def get_episode_history(self) -> List[Dict[str, Any]]:
        """Return the full action-by-action history of the current episode."""
        return list(self._history)

    # ─────────────────────────────────────────────────────────────────────────
    # Action handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _handle_classification(self, action: Action) -> bool:
        predicted = str(action.payload.get("document_type", "unknown")).lower().strip()
        if predicted not in {"invoice", "receipt"}:
            self._predicted_type = DocumentType.UNKNOWN
            return False
        self._predicted_type = DocumentType(predicted)
        return self._predicted_type == self._sample.document_type  # type: ignore[union-attr]

    def _handle_extraction(self, action: Action) -> bool:
        raw_fields = action.payload.get("fields", {})
        if not isinstance(raw_fields, dict):
            return False

        # Preserve confidence if agent provides it
        confidence_raw = raw_fields.get("confidence")
        from models.schemas import FieldConfidence  # noqa: PLC0415

        confidence = None
        if isinstance(confidence_raw, dict):
            try:
                confidence = FieldConfidence(**confidence_raw)
            except Exception:
                confidence = None

        self._extracted_fields = ExtractionFields(
            vendor_name=str(raw_fields.get("vendor_name", "")).strip() or None,
            total_amount=str(raw_fields.get("total_amount", "")).strip() or None,
            date=str(raw_fields.get("date", "")).strip() or None,
            confidence=confidence,
        )

        gt = self._sample.ground_truth  # type: ignore[union-attr]
        matches = 0
        if self._norm(self._extracted_fields.vendor_name) == self._norm(gt.vendor_name):
            matches += 1
        if self._norm_amount(self._extracted_fields.total_amount) == self._norm_amount(gt.total_amount):
            matches += 1
        if self._norm_date(self._extracted_fields.date) == self._norm_date(gt.date):
            matches += 1
        return matches >= 2

    def _handle_validation(self, action: Action) -> bool:
        vendor_ok = bool(
            self._extracted_fields.vendor_name
            and len(self._extracted_fields.vendor_name) >= 2
        )
        amount_ok = self._norm_amount(self._extracted_fields.total_amount) is not None
        date_ok = self._norm_date(self._extracted_fields.date) is not None

        env_validation = vendor_ok and amount_ok and date_ok
        claimed_valid = bool(action.payload.get("is_valid", False))

        self._validation_passed = env_validation
        return env_validation == claimed_valid

    # ─────────────────────────────────────────────────────────────────────────
    # Observation & action helpers
    # ─────────────────────────────────────────────────────────────────────────

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
                "sample_source": self._sample.source_dataset,
                "split": self._sample.split,
            },
        )

    def _available_actions(self) -> List[ActionType]:
        if self._done:
            return []
        if self._difficulty == "easy":
            return [ActionType.CLASSIFY_DOCUMENT, ActionType.FINISH]
        if self._difficulty == "medium":
            return [ActionType.EXTRACT_FIELDS, ActionType.FINISH]
        # hard — only show actions that haven't been completed yet
        actions: List[ActionType] = []
        if self._predicted_type == DocumentType.UNKNOWN:
            actions.append(ActionType.CLASSIFY_DOCUMENT)
        if self._extracted_fields.vendor_name is None:
            actions.append(ActionType.EXTRACT_FIELDS)
        if (
            self._extracted_fields.vendor_name is not None
            and not self._validation_passed
        ):
            actions.append(ActionType.VALIDATE_FIELDS)
        actions.append(ActionType.FINISH)
        return actions

    # ─────────────────────────────────────────────────────────────────────────
    # Normalisation helpers (shared with grader)
    # ─────────────────────────────────────────────────────────────────────────

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
            (r"^(\d{4})-(\d{2})-(\d{2})$", "ymd"),
            (r"^(\d{2})/(\d{2})/(\d{4})$", "dmy"),
            (r"^(\d{2})-(\d{2})-(\d{4})$", "dmy"),
        ]
        for pat, order in patterns:
            m = re.match(pat, raw)
            if not m:
                continue
            g = list(m.groups())
            if order == "ymd":
                yyyy, mm, dd = g
            else:
                dd, mm, yyyy = g
            return f"{yyyy}-{mm}-{dd}"
        return None
