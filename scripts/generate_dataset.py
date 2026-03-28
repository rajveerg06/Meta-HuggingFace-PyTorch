"""
Synthetic dataset generator for the OpenEnv Invoice/Receipt environment.

Generates realistic OCR text samples programmatically — useful for
expanding training data without downloading real Kaggle datasets.

Usage:
    python scripts/generate_dataset.py
    python scripts/generate_dataset.py --n 50 --seed 42 --output data/generated/documents.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Template data
# ─────────────────────────────────────────────────────────────────────────────

VENDORS_INVOICES = [
    "NovaTech Solutions", "Redwood Consulting", "Summit Infrastructure",
    "Pacific Trade Partners", "Horizon Legal Group", "ArcLight Media",
    "Quantum Engineering", "Atlas Construction Ltd", "Sterling Logistics",
    "BlueSky Software Inc", "Nexus Healthcare", "Orion Supplies",
    "Pinnacle Accounting", "Vertex Analytics", "Meridian Exports",
]

VENDORS_RECEIPTS = [
    "SUPERMART EXPRESS", "FRESHMART DAILY", "CITY BAKERY",
    "COFFEE HUB", "HEALTH PLUS PHARMACY", "QUICK BITE CAFE",
    "GRAND GROCER", "THE BOOKSHOP", "FAST FUEL STATION",
    "NATURE FOODS", "URBAN DELI", "TECH STORE OUTLET",
    "PET SUPPLIES WORLD", "HOMEWARE HUB", "SPORTS CORNER",
]

CURRENCIES = [
    ("$", "USD"), ("EUR", "EUR"), ("GBP", "GBP"),
    ("RM", "MYR"), ("INR", "INR"), ("AUD", "AUD"),
    ("CAD", "CAD"), ("AED", "AED"), ("SGD", "SGD"),
]

YEARS = [2022, 2023, 2024]


# ─────────────────────────────────────────────────────────────────────────────
# Generation helpers
# ─────────────────────────────────────────────────────────────────────────────


def _random_date(rng: random.Random) -> Tuple[str, str]:
    """Returns (display_date, iso_date)."""
    year = rng.choice(YEARS)
    month = rng.randint(1, 12)
    day = rng.randint(1, 28)
    fmt = rng.choice(["iso", "slash", "dash"])
    if fmt == "iso":
        return f"{year}-{month:02d}-{day:02d}", f"{year}-{month:02d}-{day:02d}"
    elif fmt == "slash":
        return f"{day:02d}/{month:02d}/{year}", f"{year}-{month:02d}-{day:02d}"
    else:
        return f"{day:02d}-{month:02d}-{year}", f"{year}-{month:02d}-{day:02d}"


def _random_amount(rng: random.Random) -> Tuple[str, str, str]:
    """Returns (currency_prefix, display_amount, numeric_str)."""
    prefix, code = rng.choice(CURRENCIES)
    dollars = rng.randint(10, 50000)
    cents = rng.choice([0, 50, 99, 25])
    numeric = f"{dollars}.{cents:02d}"
    if dollars >= 1000:
        formatted = f"{dollars:,}.{cents:02d}"
    else:
        formatted = numeric
    display = f"{prefix} {formatted}" if prefix not in {"$"} else f"${formatted}"
    return prefix, display, numeric


def _gen_invoice(rng: random.Random, difficulty: str, idx: int) -> Dict[str, Any]:
    vendor = rng.choice(VENDORS_INVOICES)
    display_date, iso_date = _random_date(rng)
    _, display_amount, numeric_amount = _random_amount(rng)
    inv_no = f"INV-{rng.randint(10000, 99999)}"
    client = rng.choice(VENDORS_INVOICES)
    while client == vendor:
        client = rng.choice(VENDORS_INVOICES)

    items = []
    n_items = rng.randint(1, 4)
    for _ in range(n_items):
        item_name = rng.choice(["Service Fee", "Product Supply", "Consulting", "Maintenance", "License"])
        items.append(item_name)
    items_text = "\n".join(f"{item}" for item in items)

    ocr = (
        f"TAX INVOICE\n"
        f"{vendor}\n"
        f"Invoice No: {inv_no}\n"
        f"Date: {display_date}\n"
        f"Bill To: {client}\n"
        f"{items_text}\n"
        f"Total Due: {display_amount}\n"
        f"Payment Terms: NET 30"
    )

    split = _split_for_idx(idx)
    return {
        "sample_id": f"gen-inv-{difficulty[:1]}-{idx:04d}",
        "source_dataset": "generated",
        "split": split,
        "difficulty": difficulty,
        "document_type": "invoice",
        "ocr_text": ocr,
        "ground_truth": {
            "vendor_name": vendor,
            "total_amount": display_amount,
            "date": display_date,
        },
    }


def _gen_receipt(rng: random.Random, difficulty: str, idx: int) -> Dict[str, Any]:
    vendor = rng.choice(VENDORS_RECEIPTS)
    display_date, iso_date = _random_date(rng)
    _, display_amount, _ = _random_amount(rng)
    receipt_no = f"R{rng.randint(10000, 99999)}"

    items = []
    n_items = rng.randint(2, 6)
    for _ in range(n_items):
        item = rng.choice(["Item A", "Item B", "Coffee", "Bread", "Milk", "Water", "Snack"])
        items.append(item)
    items_text = "\n".join(item for item in items)

    ocr = (
        f"{vendor}\n"
        f"Receipt No: {receipt_no}\n"
        f"Date: {display_date}\n"
        f"{items_text}\n"
        f"TOTAL {display_amount}\n"
        f"Thank you for shopping with us"
    )

    split = _split_for_idx(idx)
    return {
        "sample_id": f"gen-rec-{difficulty[:1]}-{idx:04d}",
        "source_dataset": "generated",
        "split": split,
        "difficulty": difficulty,
        "document_type": "receipt",
        "ocr_text": ocr,
        "ground_truth": {
            "vendor_name": vendor,
            "total_amount": display_amount,
            "date": display_date,
        },
    }


def _split_for_idx(idx: int) -> str:
    """Assign train/val/test based on index (80/10/10 split)."""
    pct = (idx % 10)
    if pct < 8:
        return "train"
    elif pct == 8:
        return "val"
    return "test"


# ─────────────────────────────────────────────────────────────────────────────
# Main generator
# ─────────────────────────────────────────────────────────────────────────────


def generate_dataset(n: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Generate N synthetic document samples.

    Args:
        n: Total number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        List of sample dictionaries in the documents.jsonl schema.
    """
    rng = random.Random(seed)
    samples: List[Dict[str, Any]] = []
    per_difficulty = n // 3
    remainder = n % 3

    difficulties = ["easy"] * per_difficulty + ["medium"] * per_difficulty + ["hard"] * (per_difficulty + remainder)

    for i, difficulty in enumerate(difficulties):
        if rng.random() < 0.5:
            sample = _gen_invoice(rng, difficulty, i)
        else:
            sample = _gen_receipt(rng, difficulty, i)
        samples.append(sample)

    return samples


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate synthetic invoice/receipt dataset for OpenEnv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n", type=int, default=30, help="Total number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default="data/generated/documents.jsonl",
        help="Output JSONL file path",
    )
    args = parser.parse_args()

    output_path = ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    samples = generate_dataset(n=args.n, seed=args.seed)

    with output_path.open("w", encoding="utf-8") as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print(f"✓ Generated {len(samples)} samples → {output_path}")
    difficulty_counts = {}
    for s in samples:
        difficulty_counts[s["difficulty"]] = difficulty_counts.get(s["difficulty"], 0) + 1
    for d, c in sorted(difficulty_counts.items()):
        print(f"  {d}: {c} samples")


if __name__ == "__main__":
    main()
