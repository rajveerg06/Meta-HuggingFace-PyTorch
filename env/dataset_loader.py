from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

from models.schemas import DocumentSample, DocumentType, ExtractionFields


class DatasetLoader:
    """
    Loads DocumentSample records from JSONL files.

    Search order (first existing path wins):
        data/samples/documents.jsonl
        data/sroie/documents.jsonl
        data/invoice/documents.jsonl
        data/rvl_cdip_subset/documents.jsonl
        data/generated/documents.jsonl

    Supports optional filtering by split ('train', 'val', 'test').
    """

    _CANDIDATE_PATHS = [
        "samples/documents.jsonl",
        "sroie/documents.jsonl",
        "invoice/documents.jsonl",
        "rvl_cdip_subset/documents.jsonl",
        "generated/documents.jsonl",
    ]

    def __init__(self, data_root: Path) -> None:
        self.data_root = Path(data_root)

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def load(
        self,
        split: Optional[str] = None,
        source: Optional[str] = None,
    ) -> List[DocumentSample]:
        """
        Load all samples, optionally filtering by split and/or source_dataset.

        Args:
            split: 'train', 'val', or 'test'; ``None`` returns all splits.
            source: dataset name filter, e.g. 'sroie'; ``None`` returns all.

        Returns:
            List of DocumentSample objects.
        """
        candidates = [self.data_root / rel for rel in self._CANDIDATE_PATHS]
        loaded_path: Optional[Path] = None
        samples: List[DocumentSample] = []

        for path in candidates:
            if path.exists():
                loaded_path = path
                samples = self._load_jsonl(path)
                break

        if loaded_path is None:
            raise FileNotFoundError(
                "No dataset found. Expected one of:\n"
                + "\n".join(f"  {self.data_root / rel}" for rel in self._CANDIDATE_PATHS)
                + "\nOr run `python scripts/generate_dataset.py` to create synthetic data."
            )

        # Optional post-load filtering
        if split is not None:
            samples = [s for s in samples if s.split == split]
        if source is not None:
            samples = [s for s in samples if s.source_dataset == source]

        if not samples:
            raise ValueError(
                f"No samples remain after filtering "
                f"(split={split!r}, source={source!r}) from {loaded_path}."
            )

        return samples

    def load_all_sources(self) -> List[DocumentSample]:
        """Load and merge from ALL existing JSONL sources (deduplicates on sample_id)."""
        seen: Dict[str, bool] = {}
        merged: List[DocumentSample] = []
        for rel in self._CANDIDATE_PATHS:
            path = self.data_root / rel
            if path.exists():
                for sample in self._load_jsonl(path):
                    if sample.sample_id not in seen:
                        seen[sample.sample_id] = True
                        merged.append(sample)
        if not merged:
            raise FileNotFoundError("No dataset files found in any candidate paths.")
        return merged

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_jsonl(path: Path) -> List[DocumentSample]:
        samples: List[DocumentSample] = []
        with path.open("r", encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    raw: Dict[str, object] = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON on line {lineno} of {path}: {exc}") from exc

                gt_raw = raw.get("ground_truth", {})
                assert isinstance(gt_raw, dict)

                sample = DocumentSample(
                    sample_id=str(raw["sample_id"]),
                    source_dataset=str(raw.get("source_dataset", "unknown")),
                    split=str(raw.get("split", "train")),
                    difficulty=str(raw["difficulty"]),  # type: ignore[arg-type]
                    ocr_text=str(raw["ocr_text"]),
                    document_type=DocumentType(str(raw["document_type"])),
                    ground_truth=ExtractionFields(
                        vendor_name=str(gt_raw.get("vendor_name", "")) or None,
                        total_amount=str(gt_raw.get("total_amount", "")) or None,
                        date=str(gt_raw.get("date", "")) or None,
                    ),
                )
                samples.append(sample)

        if not samples:
            raise ValueError(f"Dataset file {path} is empty.")
        return samples
