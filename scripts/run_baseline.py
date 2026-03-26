from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.baseline_agent import run_benchmark


if __name__ == "__main__":
    result = run_benchmark(seed=42, episodes_per_level=3)
    print(json.dumps(result, indent=2))
