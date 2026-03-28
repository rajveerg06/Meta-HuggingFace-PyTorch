"""
Enhanced baseline runner with CLI flags, export options, and formatted table output.

Usage:
    python scripts/run_baseline.py
    python scripts/run_baseline.py --agent heuristic --episodes 5 --seed 0 --export json
    python scripts/run_baseline.py --agent openai --export csv
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from io import StringIO
from pathlib import Path

# Ensure project root is on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the OpenEnv Invoice/Receipt benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--agent",
        choices=["heuristic", "openai"],
        default="heuristic",
        help="Agent to benchmark",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes per difficulty level",
    )
    parser.add_argument(
        "--export",
        choices=["none", "json", "csv"],
        default="none",
        help="Export results to file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Directory for exported result files",
    )
    return parser.parse_args()


def _format_table(result: dict) -> str:
    """Format benchmark results as an ASCII table for stdout."""
    lines = [
        "",
        "╔══════════════════════════════════════════════════════════╗",
        f"  OpenEnv Benchmark — Agent: {result['agent'].upper()}  |  seed={result['seed']}",
        "╠══════════════════════════════════════════════════════════╣",
        f"  {'DIFFICULTY':<12} {'EPISODES':>8} {'AVG SCORE':>10} {'BAR':>20}",
        "  " + "─" * 54,
    ]

    avg = result["average_score"]
    for level in ["easy", "medium", "hard"]:
        score = avg.get(level, 0.0)
        bar_len = int(score * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        count = sum(1 for ep in result["episodes"] if ep["difficulty"] == level)
        lines.append(f"  {level:<12} {count:>8} {score:>10.4f} {bar:>20}")

    lines += [
        "  " + "─" * 54,
        f"  {'OVERALL':<12} {len(result['episodes']):>8} {avg.get('overall', 0.0):>10.4f}",
        "╚══════════════════════════════════════════════════════════╝",
        "",
    ]

    # Per-episode detail
    lines += [
        "  Per-episode breakdown:",
        f"  {'ID':<22} {'DIFFICULTY':<10} {'SCORE':>8} {'REWARD':>9} {'STEPS':>6} {'MS':>8}",
        "  " + "─" * 68,
    ]
    for ep in result["episodes"]:
        lines.append(
            f"  {ep['sample_id']:<22} {ep['difficulty']:<10} "
            f"{ep['grader_score']:>8.4f} {ep['total_reward']:>9.4f} "
            f"{ep['steps_taken']:>6} {ep.get('elapsed_ms', 0.0):>7.0f}ms"
        )
    lines.append("")
    return "\n".join(lines)


def _export_json(result: dict, output_dir: str) -> None:
    path = Path(output_dir) / "benchmark_results.json"
    path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    print(f"  ✓ Results exported to: {path}")


def _export_csv(result: dict, output_dir: str) -> None:
    path = Path(output_dir) / "benchmark_results.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        if not result["episodes"]:
            return
        writer = csv.DictWriter(f, fieldnames=list(result["episodes"][0].keys()))
        writer.writeheader()
        for ep in result["episodes"]:
            writer.writerow(ep)
    print(f"  ✓ Results exported to: {path}")


def main() -> None:
    args = parse_args()

    from agent.baseline_agent import run_benchmark  # noqa: PLC0415

    print(f"\n  Running benchmark: agent={args.agent}, seed={args.seed}, episodes_per_level={args.episodes}")
    t0 = time.perf_counter()

    result_model = run_benchmark(
        seed=args.seed,
        episodes_per_level=args.episodes,
        use_openai=(args.agent == "openai"),
    )
    elapsed = time.perf_counter() - t0

    result = result_model.model_dump()
    result["total_elapsed_s"] = round(elapsed, 2)

    print(_format_table(result))

    if args.export == "json":
        _export_json(result, args.output_dir)
    elif args.export == "csv":
        _export_csv(result, args.output_dir)


if __name__ == "__main__":
    main()
