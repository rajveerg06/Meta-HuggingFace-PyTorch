from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    CLASSIFY_DOCUMENT = "classify_document"
    EXTRACT_FIELDS = "extract_fields"
    VALIDATE_FIELDS = "validate_fields"
    FINISH = "finish"


class ExtractionFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor_name: Optional[str] = None
    total_amount: Optional[str] = None
    date: Optional[str] = None


class Action(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_type: ActionType
    payload: Dict[str, Any] = Field(default_factory=dict)


class Reward(BaseModel):
    model_config = ConfigDict(extra="forbid")

    step_reward: float = 0.0
    total_reward: float = 0.0
    grader_score: float = 0.0
    details: Dict[str, float] = Field(default_factory=dict)


class Observation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    difficulty: Literal["easy", "medium", "hard"]
    ocr_text: str
    predicted_document_type: DocumentType = DocumentType.UNKNOWN
    extracted_fields: ExtractionFields = Field(default_factory=ExtractionFields)
    validation_passed: bool = False
    progress: float = 0.0
    steps_taken: int = 0
    max_steps: int = 8
    available_actions: List[ActionType] = Field(default_factory=list)
    last_action: Optional[ActionType] = None
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)


class EpisodeState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    observation: Observation
    reward: Reward


class DocumentSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source_dataset: str
    split: str = "train"
    difficulty: Literal["easy", "medium", "hard"]
    ocr_text: str
    document_type: DocumentType
    ground_truth: ExtractionFields


class ResetOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: Literal["easy", "medium", "hard"] = "easy"
    sample_index: Optional[int] = None
