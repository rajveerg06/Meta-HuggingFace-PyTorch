from __future__ import annotations

import sys
from pathlib import Path

import yaml

from env import OpenEnvInvoiceEnv
from models.schemas import Action, ActionType, Observation, Reward


def _check(condition: bool, message: str) -> None:
    if not condition:
        print(f"FAIL: {message}", file=sys.stderr)
        raise SystemExit(1)


def main() -> None:
    # ── 1. openenv.yaml structure ─────────────────────────────────────────────
    spec_path = Path("openenv.yaml")
    _check(spec_path.exists(), "openenv.yaml is missing")

    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))

    for key in ("reset", "step", "state"):
        _check(key in spec.get("api", {}), f"openenv.yaml missing api.{key}")

    for method in ("reset", "step", "state"):
        _check(
            method in spec.get("interfaces", {}).get("methods", []),
            f"openenv.yaml interfaces.methods missing '{method}'",
        )

    for ret in ("observation", "reward", "done", "info"):
        _check(
            ret in spec.get("interfaces", {}).get("step_returns", []),
            f"openenv.yaml interfaces.step_returns missing '{ret}'",
        )

    score_range = spec.get("reward", {}).get("score_range", [])
    _check(
        len(score_range) == 2 and score_range[0] == 0.0 and score_range[1] == 1.0,
        "openenv.yaml reward.score_range must be [0.0, 1.0]",
    )

    tasks = spec.get("tasks", [])
    _check(len(tasks) >= 3, "openenv.yaml must define at least 3 tasks")
    difficulties = {t.get("difficulty") for t in tasks}
    for d in ("easy", "medium", "hard"):
        _check(d in difficulties, f"openenv.yaml missing task with difficulty='{d}'")

    # ── 2. Environment contract ───────────────────────────────────────────────
    env = OpenEnvInvoiceEnv(data_root="data", seed=42)

    state = env.reset(difficulty="easy", sample_index=0)
    _check(isinstance(state.observation, Observation), "reset() observation is not Observation")

    observation, reward, done, info = env.step(
        Action(action_type=ActionType.FINISH, payload={})
    )
    _check(isinstance(observation, Observation), "step() observation is not Observation")
    _check(isinstance(reward, Reward), "step() reward is not Reward")
    _check(isinstance(done, bool), "step() done is not bool")
    _check(isinstance(info, dict), "step() info is not dict")

    _ = env.state()

    # ── 3. Reward sanity ──────────────────────────────────────────────────────
    _check(0.0 <= reward.grader_score <= 1.0, "grader_score out of [0.0, 1.0]")

    # ── 4. All three difficulties have data ───────────────────────────────────
    for diff in ("easy", "medium", "hard"):
        env2 = OpenEnvInvoiceEnv(data_root="data", seed=42)
        s = env2.reset(difficulty=diff, sample_index=0)
        _check(s.observation.difficulty == diff, f"reset difficulty mismatch for '{diff}'")

    print("prevalidate: ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
