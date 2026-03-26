from __future__ import annotations

from typing import Optional

from fastapi import FastAPI
from pydantic import BaseModel, ConfigDict

from env import OpenEnvInvoiceEnv
from models.schemas import Action, EpisodeState, ResetOptions


class ResetRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    difficulty: str = "easy"
    sample_index: Optional[int] = None


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action: Action


app = FastAPI(
    title="OpenEnv Invoice/Receipt Processing API",
    version="1.0.0",
    description="OpenEnv-compatible environment for invoice and receipt extraction tasks.",
)

env = OpenEnvInvoiceEnv(data_root="data", seed=42)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/reset", response_model=EpisodeState)
def reset(req: ResetRequest) -> EpisodeState:
    return env.reset(options=ResetOptions(difficulty=req.difficulty, sample_index=req.sample_index))


@app.post("/step", response_model=EpisodeState)
def step(req: StepRequest) -> EpisodeState:
    return env.step(req.action)


@app.get("/state", response_model=EpisodeState)
def state() -> EpisodeState:
    return env.state()
