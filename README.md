# OpenEnv Round 1 — Invoice & Receipt Understanding Environment

[![OpenEnv Compliant](https://img.shields.io/badge/OpenEnv-Round%201%20Compliant-blue)](openenv.yaml)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Problem Statement

Automated processing of financial documents (invoices and receipts) is a core
challenge in accounts-payable, procurement, and audit workflows. OCR pipelines
produce raw text with inconsistent formatting, varied date/currency conventions
and noisy layouts — making accurate field extraction non-trivial for both rule-
based systems and language models.

This environment benchmarks an agent's ability to process that raw OCR text
through a structured, multi-step pipeline: classifying the document type,
extracting key fields, and validating the result.

---

## Real-World Use Case

Enterprise finance back-offices handle thousands of invoices and receipts daily.
Manual data entry is error-prone and expensive. An LLM-powered agent that can:

1. **Classify** whether a document is an invoice or receipt
2. **Extract** vendor name, total amount, and date
3. **Validate** the extracted fields for completeness and structural correctness

…can be directly integrated into ERP systems (SAP, Oracle, QuickBooks) to
automate accounts-payable and reconciliation pipelines at scale.

---

## Environment Design

Implemented in [`env/openenv_env.py`](env/openenv_env.py) and fully compliant
with the OpenEnv Round 1 specification.

### API Contract

| Method | Signature | Returns |
|--------|-----------|---------|
| `reset()` | `difficulty`, `sample_index` | `EpisodeState` |
| `step()` | `Action` | `(Observation, Reward, bool, dict)` |
| `state()` | — | `EpisodeState` |

### Action Space (Discrete, n=4)

| Action | Description |
|--------|-------------|
| `classify_document` | Predict whether the document is `invoice` or `receipt` |
| `extract_fields` | Extract `vendor_name`, `total_amount`, `date` |
| `validate_fields` | Assert extraction completeness and correctness |
| `finish` | End episode and trigger final grader scoring |

### Observation Space

| Field | Type | Description |
|-------|------|-------------|
| `sample_id` | `str` | Unique document identifier |
| `difficulty` | `"easy"\|"medium"\|"hard"` | Episode difficulty level |
| `ocr_text` | `str` | Raw OCR text of the document |
| `predicted_document_type` | `"invoice"\|"receipt"\|"unknown"` | Agent's current classification |
| `extracted_fields` | `ExtractionFields` | `vendor_name`, `total_amount`, `date` |
| `validation_passed` | `bool` | Whether the last validation succeeded |
| `progress` | `float ∈ [0,1]` | Step progress through episode |
| `steps_taken` | `int` | Steps executed so far |
| `max_steps` | `int` | Maximum steps before timeout |
| `available_actions` | `List[str]` | Legal next actions |
| `last_action` | `Optional[str]` | Most recently executed action |
| `done` | `bool` | Whether the episode has ended |
| `info` | `dict` | Task objective and metadata |

---

## Reward & Grader Design

### Step-Level Reward Shaping

Defined in [`env/reward.py`](env/reward.py):

| Condition | Reward |
|-----------|--------|
| Correct `classify_document` | +0.25 (×1.0–1.1 difficulty boost) |
| Correct `extract_fields` (≥2/3 fields match) | +0.45 (×difficulty boost) |
| Correct `validate_fields` | +0.20 |
| `finish` action | +0.10 + **0.6 × grader_score** |
| Wrong action | −0.10 |
| Repeated action (loop) | −0.05 per repetition |
| Timeout (max_steps exceeded) | −0.10 |

### Deterministic Grader

Defined in [`graders/scoring.py`](graders/scoring.py):

| Difficulty | Formula |
|------------|---------|
| `easy` | `classification_accuracy` |
| `medium` | `0.8 × extraction_accuracy + 0.2 × completeness` |
| `hard` | `0.4 × classification_accuracy + 0.4 × extraction_accuracy + 0.2 × completeness` |

- **Vendor name**: fuzzy similarity match (≥85 → 1.0, ≥70 → 0.5, else 0.0) via `rapidfuzz`
- **Total amount**: currency-normalized float comparison with ±0.01 tolerance
- **Date**: parsed across 7 common formats, compared as `datetime.date`
- All scores clamped to `[0.0, 1.0]`

---

## Tasks

Defined in [`tasks/task_definitions.py`](tasks/task_definitions.py):

### Easy — `classify-document`

> Given OCR text of a financial document, classify whether it is an **invoice**
> or a **receipt**. Models must apply document-type reasoning over unstructured
> text containing keywords, formatting patterns, and layout cues.

Required actions: `classify_document → finish`

### Medium — `extract-key-fields`

> Extract **vendor name**, **total amount**, and **date** from the OCR text.
> Models must handle varied formatting, currency symbols, international date
> formats, and noisy OCR artifacts.

Required actions: `extract_fields → finish`

### Hard — `full-pipeline`

> Perform the complete document processing pipeline: **classify** document type,
> **extract** all key fields, and **validate** the extraction for completeness
> and structural correctness. All steps must succeed for maximum score.

Required actions: `classify_document → extract_fields → validate_fields → finish`

---

## Dataset

30 realistic synthetic documents across 3 difficulty levels (10 each), drawn
from three source datasets: `sroie`, `invoice_dataset`, `rvl_cdip_subset`.
Balanced across `train`/`val`/`test` splits.

Stored in [`data/samples/documents.jsonl`](data/samples/documents.jsonl).

---

## Setup

```bash
# 1. Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # Linux / macOS

# 2. Install dependencies
pip install -r requirements.txt

# 3. Copy and configure environment variables
copy .env.example .env          # Windows
cp .env.example .env            # Linux / macOS
# Edit .env and fill in: API_BASE_URL, MODEL_NAME, HF_TOKEN
```

---

## Run API Server

```bash
python main.py --host 0.0.0.0 --port 7860
```

Interactive docs available at: `http://localhost:7860/docs`

### Core Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode |
| `POST` | `/step` | Execute an action |
| `GET` | `/state` | Read current state (non-destructive) |
| `GET` | `/health` | Liveness check |
| `GET` | `/tasks` | List all tasks |

### Example Session

```bash
# Reset to easy episode
curl -s -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}' | python -m json.tool

# Classify the document
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify_document", "payload": {"document_type": "receipt"}}}' \
  | python -m json.tool

# Finish episode
curl -s -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "finish", "payload": {}}}' \
  | python -m json.tool
```

---

## Docker

```bash
# Build image
docker build -t openenv-invoice-receipt .

# Run container
docker run --rm -p 7860:7860 \
  -e DATA_ROOT=data \
  -e SEED=42 \
  openenv-invoice-receipt

# Health check
curl http://localhost:7860/health
```

---

## Inference

Requires environment variables `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN`.

```bash
# Set required variables
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
export HF_TOKEN="hf_..."

# Run baseline inference (reproducible, seed=42)
python inference.py --seed 42 --episodes 3 --output inference_results.json

# View summary scores
cat inference_results.json | python -m json.tool
```

The script runs 3 episodes per difficulty level (9 total) and writes a
structured JSON report to `inference_results.json`.

---

## Pre-validation

Run the compliance check before submission:

```bash
python prevalidate.py
```

Checks performed:
- `openenv.yaml` present with all required fields
- `api.reset`, `api.step`, `api.state` defined
- `interfaces.step_returns` includes `observation`, `reward`, `done`, `info`
- `reward.score_range` is `[0.0, 1.0]`
- ≥3 tasks with `easy`, `medium`, `hard` difficulties
- `reset()` returns valid `Observation` model
- `step()` returns `(Observation, Reward, bool, dict)`
- `grader_score` ∈ `[0.0, 1.0]`
- All 3 difficulty pools have samples

---

## Baseline Results

Heuristic baseline (no LLM, rule-based regex + keyword scoring):

| Difficulty | Grader Score |
|------------|-------------|
| easy | 1.0000 |
| medium | 0.9444 |
| hard | 0.9111 |
| **overall** | **0.9518** |

---

## Project Structure

```
.
├── env/                    # Core OpenEnv environment
│   ├── openenv_env.py      #   reset(), step(), state() implementation
│   ├── dataset_loader.py   #   JSONL dataset loader
│   └── reward.py           #   Step-level reward shaping
├── models/
│   └── schemas.py          # Pydantic models: Observation, Action, Reward
├── tasks/
│   └── task_definitions.py # Easy / medium / hard task definitions
├── graders/
│   └── scoring.py          # Deterministic grader (0.0–1.0)
├── api/
│   └── server.py           # FastAPI server (/reset, /step, /state)
├── agent/
│   ├── baseline_agent.py   # Heuristic agent + benchmark runner
│   └── openai_agent.py     # LLM-powered agent (OpenAI-compatible)
├── data/
│   └── samples/
│       └── documents.jsonl # 30 labelled invoice/receipt samples
├── inference.py            # Baseline inference script (OpenEnv required)
├── prevalidate.py          # Pre-submission compliance checker
├── main.py                 # Server entrypoint
├── openenv.yaml            # OpenEnv specification
├── Dockerfile              # Container build definition
├── requirements.txt        # Python dependencies
└── .env.example            # Environment variable template
```
