#!/usr/bin/env python3
import json
import argparse


def load_jsonl(path: str):
    """Load a .jsonl file into a list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main():
    """
    Build a correctness-focused APR dataset from a TSSB-style JSONL file.

    Expects each row to have:
      - "before": buggy code snippet
      - "after" : fixed / correct code snippet

    Produces a JSON file with:
      - "correct_snippets"  : list of 'after' snippets
      - "buggy_snippets"    : list of 'before' snippets
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "input",
        help="Input .jsonl file (e.g., filtered-0.jsonl, valid.jsonl, etc.)",
    )
    ap.add_argument(
        "--out",
        default="correctness_behavior_apr.json",
        help="Output .json file (default: correctness_behavior_apr.json)",
    )
    ap.add_argument(
        "--num_samples",
        type=int,
        default=0,
        help="Number of examples to keep from the start (0 = use all)",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.input)

    if args.num_samples > 0 and args.num_samples < len(rows):
        rows = rows[: args.num_samples]

    # 'after' = correct / fixed snippet
    correct_snippets = [
        ex.get("after", "").strip() for ex in rows if ex.get("after")
    ]
    # 'before' = buggy snippet
    buggy_snippets = [
        ex.get("before", "").strip() for ex in rows if ex.get("before")
    ]

    out_obj = {
        "correct_responses": correct_snippets,
        "buggy_respnses": buggy_snippets,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(
        f"Wrote {len(correct_snippets)} correct and "
        f"{len(buggy_snippets)} buggy snippets to {args.out}"
    )


if __name__ == "__main__":
    main()

