# OpenEnv Invoice/Receipt AI Environment

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.116-green.svg)](https://fastapi.tiangolo.com)
[![Pydantic](https://img.shields.io/badge/Pydantic-v2-orange.svg)](https://docs.pydantic.dev)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://docker.com)
[![Hugging Face Spaces](https://img.shields.io/badge/HF%20Spaces-Docker-yellow.svg)](https://huggingface.co/spaces)

Production-quality OpenEnv-compatible Python environment for AI-driven invoice and receipt document understanding.

---

## Problem Description

This project simulates a real-world document processing pipeline where an AI agent must:

1. **Classify** an incoming document as `invoice` or `receipt`
2. **Extract** key fields from OCR text:
   - vendor name
   - total amount (currency-agnostic, multi-format)
   - date (multi-format normalisation)
3. **Validate** extraction quality
4. **Finish** with deterministic grading and reward shaping

The environment exposes a full OpenEnv API (`reset`, `step`, `state`) backed by a FastAPI service for external evaluators. A Gymnasium-compatible metadata interface (`action_space`, `observation_space`) is also included.

---

## Project Structure

```
env/                  OpenEnv environment, dataset loading, reward logic
  openenv_env.py      Core environment (reset/step/state + render/history)
  dataset_loader.py   JSONL loader with split/source filtering
  reward.py           Reward shaping with difficulty boosts and loop penalties
  image_processor.py  Optional OCR via pytesseract (graceful fallback)

models/               Pydantic v2 models
  schemas.py          Action, Observation, Reward, EpisodeState, BenchmarkResult, ...

tasks/                Difficulty/task definitions
  task_definitions.py Easy / Medium / Hard TaskDefinition objects

graders/              Deterministic scoring
  scoring.py          Per-field scores, fuzzy vendor matching, multi-format parsing

agent/                Agents
  baseline_agent.py   Heuristic regex-based agent + benchmark runner
  openai_agent.py     LLM agent using GPT-4o-mini (requires OPENAI_API_KEY)

api/                  FastAPI server
  server.py           /reset /step /state /tasks /benchmark + more

scripts/              Utilities
  run_baseline.py     CLI benchmark runner (--agent, --seed, --episodes, --export)
  generate_dataset.py Synthetic dataset generator

tests/                Full test suite
  test_env.py         Environment lifecycle tests
  test_graders.py     Grader unit tests
  test_api.py         FastAPI integration tests

data/samples/         Bundled dataset (30 samples)
  documents.jsonl     Real-world-inspired samples across SROIE, Invoice, RVL-CDIP
```

---

## Dataset Info

The environment ships with a curated 30-sample dataset (`data/samples/documents.jsonl`) drawn from three sources:

| Source | Type | Description |
|---|---|---|
| **SROIE** | Receipts | Scanned Malaysian/Asian retail receipts |
| **Invoice Dataset** | Invoices | B2B invoices from multiple industries and currencies |
| **RVL-CDIP Subset** | Mixed | Complex document images (OCR extracted) |

### Schema

```json
{
  "sample_id": "sroie-h-001",
  "source_dataset": "sroie",
  "split": "train",
  "difficulty": "hard",
  "document_type": "receipt",
  "ocr_text": "JJ MART SDN BHD\n...",
  "ground_truth": {
    "vendor_name": "JJ MART SDN BHD",
    "total_amount": "RM 6.00",
    "date": "18/02/2024"
  }
}
```

### Supported Kaggle Datasets

Drop these into the corresponding directories and the environment auto-loads them:

| Directory | Dataset |
|---|---|
| `data/samples/` | Bundled starter dataset (used by default) |
| `data/sroie/` | [SROIE 2019](https://rrc.cvc.uab.es/?ch=13) |
| `data/invoice/` | [Invoice Dataset on Kaggle](https://www.kaggle.com/datasets) |
| `data/rvl_cdip_subset/` | [RVL-CDIP](https://adamharley.com/rvl-cdip/) |
| `data/generated/` | Output of `scripts/generate_dataset.py` |

---

## Setup

```bash
# 1. Clone / enter the project
cd Meta-HuggingFace-PyTorch

# 2. Create virtual environment
python -m venv .venv
# Activate:
.venv\Scripts\Activate.ps1   # Windows PowerShell
# source .venv/bin/activate  # Linux/macOS

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables (optional)
copy .env.example .env
# Edit .env to add OPENAI_API_KEY if using the LLM agent
```

---

## Action Space

| Action | Payload | Description |
|---|---|---|
| `classify_document` | `{"document_type": "invoice"|"receipt"}` | Classify the document type |
| `extract_fields` | `{"fields": {"vendor_name": ..., "total_amount": ..., "date": ...}}` | Extract structured fields |
| `validate_fields` | `{"is_valid": true|false}` | Validate completeness of extracted fields |
| `finish` | `{}` | End the episode and collect final reward |

---

## Observation Space

Each observation contains:

| Field | Type | Description |
|---|---|---|
| `sample_id` | str | Unique document identifier |
| `difficulty` | easy/medium/hard | Task level |
| `ocr_text` | str | Raw OCR text of the document |
| `predicted_document_type` | str | Agent's current classification |
| `extracted_fields` | object | Current extracted vendor/amount/date |
| `validation_passed` | bool | Whether validation step passed |
| `progress` | float [0,1] | Episode progress ratio |
| `steps_taken` | int | Steps taken so far |
| `max_steps` | int | Maximum steps before truncation |
| `available_actions` | list | Actions currently allowed |
| `last_action` | str | Previous action taken |
| `done` | bool | True if episode is complete |
| `info` | dict | Task objective, required actions, source, split |

---

## Difficulty Levels

| Level | Task | Grading |
|---|---|---|
| **Easy** | Classify invoice vs receipt | 100% classification accuracy |
| **Medium** | Extract key fields | 80% extraction accuracy + 20% completeness |
| **Hard** | Full pipeline: classify + extract + validate | 40% classification + 40% extraction + 20% completeness |

---

## Graders

Deterministic scoring in `[0.0, 1.0]` with per-field breakdown:

- **Vendor name**: Fuzzy matching via `rapidfuzz` (exact=1.0, ≥85 ratio=1.0, ≥70=0.5, else=0.0)
- **Total amount**: Currency-agnostic float comparison (±0.01 tolerance)
- **Date**: Multi-format normalisation (ISO, DD/MM/YYYY, DD-MM-YYYY)

---

## Reward Shaping

| Event | Reward |
|---|---|
| Correct classify | +0.25 × difficulty_boost |
| Wrong classify | -0.10 |
| Correct extract (≥2/3 fields) | +0.45 × difficulty_boost |
| Wrong extract | -0.10 |
| Validate (correctly stated) | +0.20 |
| Finish | +0.10 + 0.60 × grader_score |
| Loop penalty | -0.05 per repeated action |
| Max steps truncation | -0.10 |

---

## Running the Baseline Agent

```bash
# Heuristic agent (no API key needed)
python scripts/run_baseline.py

# With options
python scripts/run_baseline.py --agent heuristic --episodes 5 --seed 42 --export json

# OpenAI LLM agent (requires OPENAI_API_KEY in .env)
python scripts/run_baseline.py --agent openai --episodes 3
```

Example output:
```
  Running benchmark: agent=heuristic, seed=42, episodes_per_level=3

╔══════════════════════════════════════════════════════════╗
  OpenEnv Benchmark — Agent: HEURISTIC  |  seed=42
╠══════════════════════════════════════════════════════════╣
  DIFFICULTY     EPISODES  AVG SCORE                  BAR
  ──────────────────────────────────────────────────────────
  easy                  3     1.0000  ████████████████████
  medium                3     0.9444  ██████████████████░░
  hard                  3     0.9111  ██████████████████░░
  ──────────────────────────────────────────────────────────
  OVERALL               9     0.9518
╚══════════════════════════════════════════════════════════╝
```

### Baseline Results (seed=42, 3 episodes/level)

| Agent | Easy | Medium | Hard | Overall |
|---|---|---|---|---|
| Heuristic | 1.0000 | 0.9444 | 0.9111 | **0.9518** |
| OpenAI (gpt-4o-mini) | ~1.0000 | ~0.9667 | ~0.9556 | **~0.9741** |

---

## Running the API Server

```bash
python main.py
# or with options:
python main.py --port 7860 --reload
```

Interactive docs: **http://localhost:7860/docs**

### API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check + dataset stats |
| GET | `/tasks` | List all tasks with sample counts |
| GET | `/action_space` | Discrete action space descriptor |
| GET | `/observation_space` | Observation space descriptor |
| GET | `/openenv_spec` | openenv.yaml as JSON |
| POST | `/reset` | Start a new episode |
| POST | `/step` | Execute an action |
| GET | `/state` | Current episode state |
| GET | `/render` | Human-readable episode summary |
| GET | `/history` | Full episode action history |
| POST | `/benchmark` | Run a full benchmark |

### cURL Examples

```bash
# Start a hard episode
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "hard", "sample_index": 0}'

# Classify the document
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "classify_document", "payload": {"document_type": "invoice"}}}'

# Extract fields
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "extract_fields", "payload": {"fields": {"vendor_name": "Blue Ocean Logistics Pte Ltd", "total_amount": "EUR 2,190.40", "date": "2024-01-15"}}}}'

# Validate then finish
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "validate_fields", "payload": {"is_valid": true}}}'

curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action": {"action_type": "finish", "payload": {}}}'

# Get current state
curl http://localhost:7860/state

# List all tasks
curl http://localhost:7860/tasks

# Run full benchmark via API
curl -X POST http://localhost:7860/benchmark \
  -H "Content-Type: application/json" \
  -d '{"seed": 42, "episodes_per_level": 3, "agent": "heuristic"}'
```

---

## Running Tests

```bash
# Install test dependencies (included in requirements.txt)
pip install -r requirements.txt

# Run full test suite
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=. --cov-report=term-missing

# Run only environment tests
pytest tests/test_env.py -v

# Run only grader tests
pytest tests/test_graders.py -v

# Run only API tests
pytest tests/test_api.py -v
```

---

## Generating Synthetic Data

```bash
# Generate 30 samples (default)
python scripts/generate_dataset.py

# Generate 100 samples with custom seed and output
python scripts/generate_dataset.py --n 100 --seed 0 --output data/generated/documents.jsonl
```

---

## Docker

```bash
# Build
docker build -t openenv-invoice .

# Run
docker run -p 7860:7860 openenv-invoice

# Run with OpenAI key
docker run -p 7860:7860 -e OPENAI_API_KEY=sk-... openenv-invoice

# Health check
curl http://localhost:7860/health
```

---

## Hugging Face Spaces (Docker)

This repository is ready for HF Docker Spaces:

1. Create a new Space → SDK: **Docker**
2. Push all files to the Space repository
3. The space boots `uvicorn api.server:app` on port `7860`
4. Add `OPENAI_API_KEY` as a Space secret if using the LLM agent

---

## OpenEnv Spec

Full environment metadata, task definitions, action/reward schema: [`openenv.yaml`](./openenv.yaml)

---

## Optional: Image OCR Input

Install Tesseract to enable image-based input:

```bash
pip install pytesseract Pillow
# Windows: install Tesseract binary from https://github.com/UB-Mannheim/tesseract/wiki
# Ubuntu:  sudo apt-get install tesseract-ocr
```

Then use the `env.image_processor` module:

```python
from env.image_processor import ocr_from_file, is_ocr_available

if is_ocr_available():
    text = ocr_from_file("path/to/invoice.jpg")
    # Use text as ocr_text in your documents.jsonl
```
