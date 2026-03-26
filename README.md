# OpenEnv Invoice/Receipt AI Environment

Production-oriented OpenEnv-compatible Python environment for invoice and receipt document understanding.

## Problem Description

This project simulates real-world document processing workflows where an AI agent must:

1. Classify an incoming document as `invoice` or `receipt`
2. Extract key fields from OCR text:
   - vendor name
   - total amount
   - date
3. Validate extraction quality
4. Complete the task with deterministic grading and reward shaping

The environment exposes OpenEnv-like APIs (`reset`, `step`, `state`) and also a FastAPI service for external evaluators.

## Dataset Info

The environment supports local Kaggle-derived samples from:

- SROIE receipt dataset
- Invoice datasets
- RVL-CDIP subset

Expected data files (first existing path is used):

- `data/samples/documents.jsonl`
- `data/sroie/documents.jsonl`
- `data/invoice/documents.jsonl`
- `data/rvl_cdip_subset/documents.jsonl`

A realistic starter dataset is included in `data/samples/documents.jsonl` using the same schema expected for local simulation.

### JSONL Schema

```json
{
  "sample_id": "string",
  "source_dataset": "sroie|invoice_dataset|rvl_cdip_subset",
  "split": "train|val|test",
  "difficulty": "easy|medium|hard",
  "document_type": "invoice|receipt",
  "ocr_text": "raw OCR text",
  "ground_truth": {
    "vendor_name": "string",
    "total_amount": "string",
    "date": "string"
  }
}
```

## Architecture

```text
env/       -> OpenEnv environment, dataset loading, reward logic
models/    -> Pydantic models for observation/action/reward/state
tasks/     -> difficulty/task definitions
graders/   -> deterministic scoring
agent/     -> baseline heuristic agent and benchmark runner
api/       -> FastAPI server exposing reset/step/state
```

## Action Space

- `classify_document`
- `extract_fields`
- `validate_fields`
- `finish`

## Observation Space

Each observation contains:

- OCR text
- current extracted fields
- predicted doc type
- task progress
- available actions
- done flag and task metadata

## Difficulty Levels

1. `easy`: classify invoice vs receipt
2. `medium`: extract key fields
3. `hard`: full pipeline classify + extract + validate

## Deterministic Graders

Scores are deterministic and bounded in `[0.0, 1.0]`:

- classification accuracy
- field extraction accuracy
- completeness

Weighted per difficulty:

- easy: classification only
- medium: extraction + completeness
- hard: classification + extraction + completeness

## Reward Shaping

- Partial positive rewards for correct intermediate actions
- Negative rewards for incorrect actions
- Loop penalties for repeated actions
- Final reward bonus based on grader score

## Setup

```bash
python -m venv .venv
. .venv/Scripts/activate  # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Run Baseline Agent

```bash
python scripts/run_baseline.py
```

This executes all task levels with fixed seed for reproducible scores.

## Run API Server

```bash
python main.py
```

Endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /health`

Example reset payload:

```json
{
  "difficulty": "hard",
  "sample_index": 0
}
```

## Docker

```bash
docker build -t openenv-invoice .
docker run -p 7860:7860 openenv-invoice
```

## Hugging Face Spaces (Docker)

This repository is ready for Docker Spaces:

1. Create a new Space with SDK = Docker
2. Push this repository content
3. Space will run `uvicorn api.server:app` on port `7860`

## OpenEnv Spec

Environment metadata and tasks are defined in `openenv.yaml`.

## Baseline Results

Baseline scores are generated dynamically by:

```bash
python scripts/run_baseline.py
```

Because the environment seed and sample order are fixed, results are reproducible across runs.

Example output with default settings (`seed=42`, `episodes_per_level=3`):

- easy: `1.0000`
- medium: `1.0000`
- hard: `0.9556`
- overall: `0.9852`
