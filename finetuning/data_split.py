#!/usr/bin/env python3
"""
Stratified train/valid split for APR dataset by bug type (sstub_pattern),
with sanity check on distribution.
"""

import argparse, json, random
from collections import defaultdict, Counter

def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def stratified_split(rows, val_ratio=0.1, key="sstub_pattern", seed=42):
    buckets = defaultdict(list)
    for r in rows:
        buckets[r.get(key, "UNK")].append(r)

    random.seed(seed)
    train, valid = [], []
    for k, items in buckets.items():
        random.shuffle(items)
        cut = int(len(items) * val_ratio)
        valid.extend(items[:cut])
        train.extend(items[cut:])
    return train, valid

def report_distribution(rows, label, key="sstub_pattern", topn=10):
    counts = Counter(r.get(key, "UNK") for r in rows)
    total = len(rows)
    print(f"\n[{label}] n={total}")
    for bug, cnt in counts.most_common(topn):
        print(f"  {bug:20s} {cnt:5d} ({cnt/total:.2%})")
    if len(counts) > topn:
        print(f"  ... ({len(counts)-topn} more bug types)")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="all_data.jsonl")
    ap.add_argument("--train_out", default="train.jsonl")
    ap.add_argument("--valid_out", default="valid.jsonl")
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = read_jsonl(args.input)
    print(f"Loaded {len(rows)} examples")

    train, valid = stratified_split(rows, args.val_ratio, "sstub_pattern", args.seed)
    print(f"\nSplit sizes: Train={len(train)} | Valid={len(valid)}")

    # Write files
    write_jsonl(args.train_out, train)
    write_jsonl(args.valid_out, valid)
    print(f"Saved train to {args.train_out}, valid to {args.valid_out}")

    # Sanity check distributions
    report_distribution(train, "TRAIN", "sstub_pattern")
    report_distribution(valid, "VALID", "sstub_pattern")

if __name__ == "__main__":
    main()

