from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Optional

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

from env import OpenEnvInvoiceEnv
from models.schemas import (
    Action,
    BenchmarkResult,
    EpisodeState,
    ResetOptions,
    TaskInfo,
)
from tasks import TASK_DEFINITIONS

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Application lifecycle
# ─────────────────────────────────────────────────────────────────────────────

_env: Optional[OpenEnvInvoiceEnv] = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _env
    data_root = os.getenv("DATA_ROOT", "data")
    seed = int(os.getenv("SEED", "42"))
    logger.info("Loading dataset from %s (seed=%d)…", data_root, seed)
    _env = OpenEnvInvoiceEnv(data_root=data_root, seed=seed)
    logger.info("Dataset loaded. %d samples across all difficulties.", len(_env.dataset))
    yield
    logger.info("Shutting down OpenEnv environment.")


def _get_env() -> OpenEnvInvoiceEnv:
    if _env is None:
        raise RuntimeError("Environment not initialised. Wait for startup to complete.")
    return _env


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OpenEnv Invoice/Receipt Processing API",
    version="1.1.0",
    description=(
        "OpenEnv-compatible environment for invoice and receipt extraction tasks. "
        "Implements `reset`, `step`, `state` lifecycle endpoints plus benchmark and task discovery."
    ),
    lifespan=lifespan,
)

# CORS — required for Hugging Face Spaces iframe embedding and external evaluators
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Request models
# ─────────────────────────────────────────────────────────────────────────────


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: str = "easy"
    sample_index: Optional[int] = None
    split: Optional[str] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action


class BenchmarkRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    seed: int = 42
    episodes_per_level: int = 3
    agent: str = "heuristic"  # "heuristic" | "openai"


# ─────────────────────────────────────────────────────────────────────────────
# Health & meta endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Meta"])
def health() -> Dict[str, Any]:
    """Liveness check — returns dataset size and env status."""
    env = _get_env()
    return {
        "status": "ok",
        "total_samples": len(env.dataset),
        "samples_by_difficulty": {
            d: len(samples) for d, samples in env.samples_by_difficulty.items()
        },
        "action_space": env.action_space,
        "fuzzy_matching": True,
    }


@app.get("/tasks", response_model=list[TaskInfo], tags=["Meta"])
def list_tasks() -> list[TaskInfo]:
    """List all available tasks with difficulty levels and sample counts."""
    env = _get_env()
    result = []
    for difficulty, task in TASK_DEFINITIONS.items():
        result.append(
            TaskInfo(
                task_id=task.task_id,
                difficulty=task.difficulty,  # type: ignore[arg-type]
                objective=task.objective,
                required_actions=[a.value for a in task.required_actions],
                sample_count=len(env.samples_by_difficulty.get(difficulty, [])),
            )
        )
    return result


@app.get("/action_space", tags=["Meta"])
def action_space() -> Dict[str, Any]:
    """Return the discrete action space descriptor."""
    return _get_env().action_space


@app.get("/observation_space", tags=["Meta"])
def observation_space() -> Dict[str, Any]:
    """Return the observation space descriptor."""
    return _get_env().observation_space


@app.get("/openenv_spec", tags=["Meta"])
def openenv_spec() -> Dict[str, Any]:
    """Return the openenv.yaml specification as JSON."""
    try:
        with open("openenv.yaml", "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="openenv.yaml not found")


# ─────────────────────────────────────────────────────────────────────────────
# Core OpenEnv endpoints
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/reset", response_model=EpisodeState, tags=["OpenEnv"])
def reset(req: ResetRequest) -> EpisodeState:
    """
    Reset the environment to start a new episode.

    Body:
        difficulty: 'easy' | 'medium' | 'hard'  (default: 'easy')
        sample_index: optional integer to pin to a specific sample
        split: 'train' | 'val' | 'test' | null
    """
    try:
        return _get_env().reset(
            options=ResetOptions(
                difficulty=req.difficulty,  # type: ignore[arg-type]
                sample_index=req.sample_index,
                split=req.split,  # type: ignore[arg-type]
            )
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/step", response_model=EpisodeState, tags=["OpenEnv"])
def step(req: StepRequest) -> EpisodeState:
    """
    Execute an action in the current episode.

    Body:
        action.action_type: 'classify_document' | 'extract_fields' | 'validate_fields' | 'finish'
        action.payload: action-specific parameters
    """
    try:
        return _get_env().step_state(req.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/state", response_model=EpisodeState, tags=["OpenEnv"])
def state() -> EpisodeState:
    """Return the current episode state without advancing the environment."""
    try:
        return _get_env().state()
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/render", tags=["OpenEnv"])
def render() -> Dict[str, str]:
    """Return a human-readable text summary of the current episode state."""
    try:
        return {"render": _get_env().render()}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


@app.get("/history", tags=["OpenEnv"])
def history() -> Dict[str, Any]:
    """Return the full action-by-action history of the current episode."""
    try:
        return {"history": _get_env().get_episode_history()}
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark endpoint
# ─────────────────────────────────────────────────────────────────────────────


@app.post("/benchmark", response_model=BenchmarkResult, tags=["Benchmark"])
def benchmark(req: BenchmarkRequest) -> BenchmarkResult:
    """
    Run a full benchmark across all difficulty levels.

    This creates a temporary environment instance so as NOT to disturb
    the shared env state used by /reset → /step → /state flows.

    Body:
        seed: random seed (default: 42)
        episodes_per_level: episodes to run per difficulty (default: 3)
        agent: 'heuristic' | 'openai'
    """
    if req.agent not in {"heuristic", "openai"}:
        raise HTTPException(status_code=400, detail="agent must be 'heuristic' or 'openai'")

    from agent.baseline_agent import run_benchmark  # noqa: PLC0415

    try:
        return run_benchmark(
            seed=req.seed,
            episodes_per_level=req.episodes_per_level,
            use_openai=(req.agent == "openai"),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
