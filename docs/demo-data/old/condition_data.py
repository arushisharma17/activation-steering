#!/usr/bin/env python3
# Build a train/test prompt dataset keyed by fixed bug types.
# Each bug-type field uses:
# "Fix the following buggy snippet. Return only the corrected code, with no extra commentary."
# followed by the buggy snippet and "Answer:".

import json
import argparse
import random
from collections import defaultdict
from typing import List, Dict, Any, Iterable

# Fixed set of bug-type keys (exact names & order you provided)
BUGTYPE_KEYS = [
    "MORE_SPECIFIC_IF",
    "ADD_METHOD_CALL",
    "ADD_FUNCTION_AROUND_EXPRESSION",
    "SAME_FUNCTION_MORE_ARGS",
    "SAME_FUNCTION_LESS_ARGS",
    "CHANGE_BINARY_OPERATOR",
    "CHANGE_COMPARISON_OPERATOR",
    "SINGLE_TOKEN",
]

def load_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows

def bucket_by_bugtype(rows: Iterable[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    buckets = defaultdict(list)
    for ex in rows:
        bt = ex.get("sstub_pattern")
        if not bt:
            continue
        # keep examples that have both sides
        if ex.get("before") and ex.get("after"):
            buckets[bt].append(ex)
    return buckets

def prompt_for_example(ex: Dict[str, Any]) -> str:
    buggy = str(ex.get("before", "")).strip()
    return (
        "Fix the following buggy snippet. Return only the corrected code, with no extra commentary.\n\n"
        "Buggy:\n"
        f"{buggy}\n\n"
        "Answer: "
    )

def build_rows_mosaic(
    buckets: Dict[str, List[Dict[str, Any]]],
    keys_in_use: List[str],
    limit_rows: int,
    seed: int
) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    # Shuffle each bucket for decorrelation
    shuf = {}
    max_len = 0
    for bt in keys_in_use:
        arr = list(buckets.get(bt, []))
        rng.shuffle(arr)
        shuf[bt] = arr
        max_len = max(max_len, len(arr))

    if max_len == 0:
        return []

    n_rows = max_len if limit_rows <= 0 else min(limit_rows, max_len)
    rows = []
    for i in range(n_rows):
        row = {
            "base": (
                "Summarize the minimal code change needed to fix a small bug. "
                "Respond briefly in one or two sentences."
            )
        }
        for bt in keys_in_use:
            arr = shuf[bt]
            if not arr:
                continue
            ex = arr[i % len(arr)]
            row[bt] = prompt_for_example(ex)
        rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser(description="Build train/test prompt dataset keyed by fixed bug types.")
    ap.add_argument("input", help="Input bug JSONL (with before/after/sstub_pattern, etc.)")
    ap.add_argument("--out", default="bugtype_dataset.json", help="Output JSON path.")
    ap.add_argument("--train_ratio", type=float, default=0.9, help="Train split ratio.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed.")
    ap.add_argument("--limit_rows", type=int, default=0, help="Cap the number of rows (0 = auto).")
    ap.add_argument("--min_per_type", type=int, default=1,
                    help="Drop bug types with fewer than this many examples.")
    ap.add_argument("--strict_types", action="store_true",
                    help="Require all 8 bug types to be present with >= min_per_type examples (else exit).")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    buckets_all = bucket_by_bugtype(rows)

    # Filter to the fixed set, applying min_per_type
    keys_in_use = []
    for bt in BUGTYPE_KEYS:
        n = len(buckets_all.get(bt, []))
        if n >= args.min_per_type:
            keys_in_use.append(bt)

    if args.strict_types and len(keys_in_use) != len(BUGTYPE_KEYS):
        missing = [bt for bt in BUGTYPE_KEYS if bt not in keys_in_use]
        raise SystemExit(f"Strict mode: missing or underfilled types: {missing}")

    if not keys_in_use:
        raise SystemExit("No bug types available with the current constraints.")

    mosaic_rows = build_rows_mosaic(buckets_all, keys_in_use, args.limit_rows, args.seed)
    if not mosaic_rows:
        raise SystemExit("No rows could be built (check data and filters).")

    rng = random.Random(args.seed)
    rng.shuffle(mosaic_rows)
    n = len(mosaic_rows)
    k = int(round(args.train_ratio * n))
    dataset = {"train": mosaic_rows[:k], "test": mosaic_rows[k:]}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(dataset['train'])} train and {len(dataset['test'])} test rows to {args.out}")
    print("Bug types included (with counts):")
    for bt in BUGTYPE_KEYS:
        print(f"  {bt}: {len(buckets_all.get(bt, []))} "
              f"{'' if bt in keys_in_use else '(dropped)'}")

if __name__ == "__main__":
    main()

