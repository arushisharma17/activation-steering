#!/usr/bin/env python3
import json
import argparse
import random

# A pool of varied templates; {buggy} and {pattern} will be filled in.
TEMPLATES = [
    "Given the buggy code: `{buggy}`, rewrite it into a correct version. (Bug type: {pattern})",
    "Fix the following statement so it is correct: `{buggy}` (Pattern: {pattern})",
    "Identify the issue in `{buggy}` and provide the corrected single-line fix. [{pattern}]",
    "Produce the corrected version of: `{buggy}`  (family: {pattern})",
    "This line is buggy: `{buggy}`. Provide the correct version only. [{pattern}]",
    "Apply the minimal change that makes this line correct: `{buggy}`  ({pattern})",
    "Replace the buggy snippet with a correct one: `{buggy}`  [pattern={pattern}]",
    "Correct the following line while preserving intent: `{buggy}`  ({pattern})",
    "Fix the error in: `{buggy}`. Return only the corrected line. [{pattern}]",
]

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            rows.append(json.loads(s))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Create APR-style questions from buggy code.")
    ap.add_argument("input", help="Input .jsonl with fields including 'before' and 'sstub_pattern'")
    ap.add_argument("--out", default="questions.json", help="Output JSON file (with {'train': [...]})")
    ap.add_argument("--start", type=int, default=0, help="Start index (inclusive, 0-based)")
    ap.add_argument("--end", type=int, default=0, help="End index (exclusive, 0 = to end)")
    ap.add_argument("--seed", type=int, default=17, help="Random seed for template selection")
    args = ap.parse_args()

    rows = load_jsonl(args.input)
    n = len(rows)
    end = args.end if args.end > 0 else n

    # slice
    subset = rows[args.start:end]

    random.seed(args.seed)

    questions = []
    for ex in subset:
        buggy = (ex.get("before") or "").strip()
        if not buggy:
            continue
        pattern = (ex.get("sstub_pattern") or "bug").strip()

        # choose a template at random
        tpl = random.choice(TEMPLATES)

        # escape backticks inside buggy snippet
        safe_buggy = buggy.replace("`", "\\`")

        q = tpl.format(buggy=safe_buggy, pattern=pattern)
        questions.append({"question": q})

    out = {"train": questions}

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(questions)} questions to {args.out} (from indices {args.start} to {end} of {n})")

if __name__ == "__main__":
    main()

