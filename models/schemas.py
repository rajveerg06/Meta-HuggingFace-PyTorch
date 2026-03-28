from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────────────
# Enumerations
# ─────────────────────────────────────────────────────────────────────────────


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class ActionType(str, Enum):
    CLASSIFY_DOCUMENT = "classify_document"
    EXTRACT_FIELDS = "extract_fields"
    VALIDATE_FIELDS = "validate_fields"
    FINISH = "finish"


# ─────────────────────────────────────────────────────────────────────────────
# Field-level models
# ─────────────────────────────────────────────────────────────────────────────


class FieldConfidence(BaseModel):
    """Per-field confidence scores produced by LLM agents (0.0–1.0)."""

    model_config = ConfigDict(extra="forbid")

    vendor_name: float = Field(default=0.0, ge=0.0, le=1.0)
    total_amount: float = Field(default=0.0, ge=0.0, le=1.0)
    date: float = Field(default=0.0, ge=0.0, le=1.0)


class ExtractionFields(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vendor_name: Optional[str] = None
    total_amount: Optional[str] = None
    date: Optional[str] = None
    # Optional per-field confidence (populated by LLM agents; ignored by grader)
    confidence: Optional[FieldConfidence] = None


class FieldScoreDetail(BaseModel):
    """Per-field grading breakdown returned inside Reward.details."""

    model_config = ConfigDict(extra="forbid")

    vendor_name_score: float = 0.0
    total_amount_score: float = 0.0
    date_score: float = 0.0
    classification_accuracy: float = 0.0
    extraction_accuracy: float = 0.0
    completeness: float = 0.0
    final_score: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Image input (optional — for future OCR pipeline support)
# ─────────────────────────────────────────────────────────────────────────────


class ImageInput(BaseModel):
    """Optional image-based input. Either file_path *or* base64_data must be provided."""

    model_config = ConfigDict(extra="forbid")

    file_path: Optional[str] = Field(default=None, description="Absolute or relative path to image file")
    base64_data: Optional[str] = Field(default=None, description="Base64-encoded image content")
    mime_type: str = Field(default="image/jpeg", description="MIME type of the image")


# ─────────────────────────────────────────────────────────────────────────────
# Core interaction models
# ─────────────────────────────────────────────────────────────────────────────


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


# ─────────────────────────────────────────────────────────────────────────────
# Dataset models
# ─────────────────────────────────────────────────────────────────────────────


class DocumentSample(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_id: str
    source_dataset: str
    split: str = "train"
    difficulty: Literal["easy", "medium", "hard"]
    ocr_text: str
    document_type: DocumentType
    ground_truth: ExtractionFields


# ─────────────────────────────────────────────────────────────────────────────
# API request/response models
# ─────────────────────────────────────────────────────────────────────────────


class ResetOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: Literal["easy", "medium", "hard"] = "easy"
    sample_index: Optional[int] = None
    split: Optional[Literal["train", "val", "test"]] = None


class TaskInfo(BaseModel):
    """Returned by GET /tasks."""

    model_config = ConfigDict(extra="forbid")

    task_id: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    required_actions: List[str]
    sample_count: int = 0


class EpisodeRecord(BaseModel):
    """Single episode result for benchmark reports."""

    model_config = ConfigDict(extra="forbid")

    difficulty: str
    sample_id: str
    total_reward: float
    grader_score: float
    steps_taken: int
    elapsed_ms: float


class BenchmarkResult(BaseModel):
    """Full benchmark run result returned by /benchmark and the CLI."""

    model_config = ConfigDict(extra="forbid")

    agent: str
    seed: int
    episodes_per_level: int
    episodes: List[EpisodeRecord]
    average_score: Dict[str, float]
