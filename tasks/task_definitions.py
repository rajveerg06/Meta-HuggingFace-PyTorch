from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from models.schemas import ActionType


@dataclass(frozen=True)
class TaskDefinition:
    task_id: str
    difficulty: str
    objective: str
    required_actions: List[ActionType]


TASK_DEFINITIONS: Dict[str, TaskDefinition] = {
    "easy": TaskDefinition(
        task_id="classify-document",
        difficulty="easy",
        objective="Classify whether the input document is an invoice or receipt.",
        required_actions=[ActionType.CLASSIFY_DOCUMENT, ActionType.FINISH],
    ),
    "medium": TaskDefinition(
        task_id="extract-key-fields",
        difficulty="medium",
        objective="Extract vendor name, total amount, and date from OCR text.",
        required_actions=[ActionType.EXTRACT_FIELDS, ActionType.FINISH],
    ),
    "hard": TaskDefinition(
        task_id="full-pipeline",
        difficulty="hard",
        objective="Classify document, extract key fields, and validate the extraction.",
        required_actions=[
            ActionType.CLASSIFY_DOCUMENT,
            ActionType.EXTRACT_FIELDS,
            ActionType.VALIDATE_FIELDS,
            ActionType.FINISH,
        ],
    ),
}
