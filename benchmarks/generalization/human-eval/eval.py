#!/usr/bin/env python3
"""
Evaluate HumanEval generation files in results/ and write a summary CSV.

This script:
  - Scans results/ for JSONL files whose names contain "_humaneval_".
  - Skips files that look like evaluation artifacts (contain "summary" or "results" in name).
  - Computes pass@1, pass@5, pass@10 for each generation file.
  - Writes a single combined summary CSV to results/.

Filename parsing is intentionally generic and does not assume any specific dataset names.
It attempts to extract:
  - model slug
  - n (samples per task) from token like "n10"
  - steered flag if "steered" appears in the filename tokens
  - an optional free-form "tag" token (if present) used only for grouping/reporting
"""

import argparse
import os
import glob
import datetime
import pandas as pd
from human_eval.evaluation import evaluate_functional_correctness


def parse_filename(path: str):
    """
    Parse a HumanEval generation filename to extract:

      - model slug
      - n (samples per task, if present)
      - steered (bool)
      - tag (str): optional free-form token used only for reporting ("none" if absent)

    Expected stem patterns (filename without ".jsonl"):

      <slug>_humaneval_n10
      <slug>_humaneval_n10_baseline
      <slug>_humaneval_n10_steered
      <slug>_humaneval_n10_<tag>_steered_<condition...>
      <slug>_humaneval_n10_<tag>_<condition...>

    Returns:
      (model_slug: str, n: int|None, steered: bool, tag: str)
    """
    fname = os.path.basename(path)
    stem = fname[:-6] if fname.endswith(".jsonl") else fname

    if "_humaneval_" not in stem:
        return stem, None, False, "none"

    slug, rest = stem.split("_humaneval_", 1)
    parts = rest.split("_")

    # Find n from token like "n10"
    n = None
    idx_n = None
    for i, p in enumerate(parts):
        if p.startswith("n") and p[1:].isdigit():
            idx_n = i
            n = int(p[1:])
            break

    # Defaults
    steered = False
    tag = "none"

    if idx_n is None:
        return slug, n, steered, tag

    tail = parts[idx_n + 1 :]

    # Determine steering based on any token containing "steered"
    if any("steered" in t for t in tail):
        steered = True

    # Heuristic for "tag":
    # If the first token after n is present and is not a common setting token,
    # treat it as a tag used only for reporting.
    if tail:
        first = tail[0]
        if first not in ("baseline", "steered"):
            tag = first

    return slug, n, steered, tag


def main():
    ap = argparse.ArgumentParser(description="Evaluate all HumanEval generation files in results/.")

    ap.add_argument("--results_dir", default="results", help="Directory containing HumanEval JSONL files.")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers for evaluation.")
    ap.add_argument("--timeout", type=float, default=5.0, help="Per-test timeout in seconds.")
    ap.add_argument("--ks", default="1,5,10", help="Comma-separated k values for pass@k (e.g., '1,5,10').")

    args = ap.parse_args()

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)

    try:
        ks = [int(x.strip()) for x in args.ks.split(",") if x.strip()]
        if not ks:
            raise ValueError
    except Exception:
        raise ValueError(f"Invalid --ks value: {args.ks!r}. Example: --ks 1,5,10")

    # Collect generation files
    all_jsonl = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    jsonl_files = []
    for path in all_jsonl:
        fname = os.path.basename(path)
        if "_humaneval_" not in fname:
            continue
        # Skip likely evaluation artifacts
        lowered = fname.lower()
        if "summary" in lowered or "results" in lowered:
            continue
        jsonl_files.append(path)

    if not jsonl_files:
        print(f"[WARN] No HumanEval JSONL files found in {results_dir}/. Did you run gen.py?")
        return

    print(f"[INFO] Found {len(jsonl_files)} HumanEval generation files to evaluate.")

    rows = []

    for path in jsonl_files:
        slug, n, steered, tag = parse_filename(path)
        print(f"[INFO] Evaluating {os.path.basename(path)} (model={slug}, n={n}, steered={steered}, tag={tag})")

        metrics = evaluate_functional_correctness(
            sample_file=path,
            k=ks,
            n_workers=args.workers,
            timeout=args.timeout,
        )

        row = {
            "file": os.path.basename(path),
            "model": slug,
            "n": n,
            "steered": steered,
            "tag": tag,
            "n_samples": metrics.get("n_samples") or metrics.get("n_total"),
        }
        for k in ks:
            row[f"pass@{k}"] = metrics.get(f"pass@{k}")

        rows.append(row)

    if not rows:
        print("[WARN] No rows collected; nothing to write.")
        return

    df = pd.DataFrame(rows)

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(results_dir, f"humaneval_summary_{ts}.csv")
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Wrote summary to {out_csv}")


if __name__ == "__main__":
    main()

