from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from models.schemas import DocumentSample, DocumentType, ExtractionFields


class DatasetLoader:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root

    def load(self) -> List[DocumentSample]:
        candidates = [
            self.data_root / "samples" / "documents.jsonl",
            self.data_root / "sroie" / "documents.jsonl",
            self.data_root / "invoice" / "documents.jsonl",
            self.data_root / "rvl_cdip_subset" / "documents.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return self._load_jsonl(path)
        raise FileNotFoundError(
            "No dataset found. Place a Kaggle-derived documents.jsonl under data/sroie, data/invoice, "
            "or data/rvl_cdip_subset; or use data/samples/documents.jsonl."
        )

    def _load_jsonl(self, path: Path) -> List[DocumentSample]:
        samples: List[DocumentSample] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                raw: Dict[str, object] = json.loads(line)
                ground_truth = raw.get("ground_truth", {})
                sample = DocumentSample(
                    sample_id=str(raw["sample_id"]),
                    source_dataset=str(raw.get("source_dataset", "unknown")),
                    split=str(raw.get("split", "train")),
                    difficulty=str(raw["difficulty"]),
                    ocr_text=str(raw["ocr_text"]),
                    document_type=DocumentType(str(raw["document_type"])),
                    ground_truth=ExtractionFields(
                        vendor_name=str(ground_truth.get("vendor_name", "")),
                        total_amount=str(ground_truth.get("total_amount", "")),
                        date=str(ground_truth.get("date", "")),
                    ),
                )
                samples.append(sample)
        if not samples:
            raise ValueError(f"Dataset file {path} is empty.")
        return samples
