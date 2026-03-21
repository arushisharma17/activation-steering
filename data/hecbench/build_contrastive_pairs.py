#!/usr/bin/env python3
"""
Convert raw LLM bug-injection output into contrastive pairs JSON.

Reads raw_bugs.jsonl and produces a JSON file matching the format used
elsewhere in the activation-steering project:

    {
      "correct_responses": ["<original code 1>", ...],
      "buggy_responses":   ["<buggy code 1>",   ...]
    }

Optionally splits output by bug category (--by-category).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from config import CONTRASTIVE_PAIRS_PATH, RAW_BUGS_PATH


def load_raw(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def build_pairs(rows: list[dict]) -> dict:
    """Build the contrastive pair structure."""
    correct = []
    buggy = []
    for row in rows:
        orig = row.get("original_code", "").strip()
        bug = row.get("buggy_code", "").strip()
        if orig and bug:
            correct.append(orig)
            buggy.append(bug)
    return {
        "correct_responses": correct,
        "buggy_responses": buggy,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Build contrastive pairs from raw bugs")
    ap.add_argument("--input", type=Path, default=RAW_BUGS_PATH)
    ap.add_argument("--out", type=Path, default=CONTRASTIVE_PAIRS_PATH)
    ap.add_argument(
        "--by-category",
        action="store_true",
        help="Also write per-bug-type JSON files",
    )
    args = ap.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input file not found: {args.input}")
        print("       Run inject_bugs_llm.py first.")
        sys.exit(1)

    rows = load_raw(args.input)
    print(f"[INFO] Loaded {len(rows)} raw rows from {args.input}")

    # Explicitly filter out passed files
    rows = [r for r in rows if r.get("bug_type") != "PASS"]
    print(f"[INFO] Kept {len(rows)} valid bug injection rows")

    # --- Combined output ---
    pairs = build_pairs(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(pairs, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"[DONE] Wrote {len(pairs['correct_responses'])} pairs to {args.out}"
    )

    # --- Per-category ---
    if args.by_category:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for row in rows:
            by_cat[row.get("bug_type", "UNKNOWN")].append(row)

        for cat, cat_rows in sorted(by_cat.items()):
            cat_pairs = build_pairs(cat_rows)
            cat_path = args.out.parent / f"contrastive_pairs_{cat.lower()}.json"
            cat_path.write_text(
                json.dumps(cat_pairs, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"  → {cat}: {len(cat_pairs['correct_responses'])} pairs → {cat_path.name}")


if __name__ == "__main__":
    main()
