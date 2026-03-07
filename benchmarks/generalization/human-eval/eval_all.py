#!/usr/bin/env python3
"""
Evaluate all HumanEval generation files in results/ and write summary CSVs.

Supported filename patterns (JSONL in results/):

Legacy (no dataset tag):
  <slug>_humaneval_n10.jsonl
  <slug>_humaneval_n10_steered.jsonl

New (with dataset + setting):
  <slug>_humaneval_n10_<dataset>_<setting>.jsonl
    e.g., Qwen2.5-7B-Instruct_humaneval_n10_tssb_steered_correctness-tssb.jsonl
          Qwen2.5-7B-Instruct_humaneval_n10_manysstubs_baseline.jsonl

This script:
  - Skips any files whose names contain "results" (evaluation artifacts).
  - Computes pass@1, pass@5, pass@10 for each generation file.
  - Writes:
      results/humaneval_summary_all_<timestamp>.csv
    and per-dataset summaries:
      results/<dataset>/humaneval_summary_<dataset>_<timestamp>.csv

Where <dataset> is one of:
  - "tssb"
  - "manysstubs"
  - "none"   (legacy runs or no dataset tag found)
"""

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
      - dataset tag (str), e.g. "tssb", "manysstubs", or "none"

    Expected patterns (stem = filename without ".jsonl"):

      Legacy:
        <slug>_humaneval_n10
        <slug>_humaneval_n10_steered

      New:
        <slug>_humaneval_n10_<dataset>_<setting>

    Returns:
      (model_slug: str, n: int or None, steered: bool, dataset: str)
    """
    fname = os.path.basename(path)
    stem = fname[:-6] if fname.endswith(".jsonl") else fname  # drop ".jsonl"

    if "_humaneval_" not in stem:
        # Not in expected format; treat as generic, dataset="none"
        return stem, None, False, "none"

    slug, rest = stem.split("_humaneval_", 1)
    parts = rest.split("_")

    # Find n from something like "n10"
    n = None
    idx_n = None
    for i, p in enumerate(parts):
        if p.startswith("n") and p[1:].isdigit():
            idx_n = i
            try:
                n = int(p[1:])
            except ValueError:
                n = None
            break

    # Default dataset + steering flags
    dataset = "none"
    steered = False

    if idx_n is None:
        # No "n<k>" token; treat as legacy / unknown
        return slug, n, steered, dataset

    tail = parts[idx_n + 1 :]

    if not tail:
        # Pattern: <slug>_humaneval_n10
        dataset = "none"
        steered = False
    elif len(tail) == 1:
        # Pattern: <slug>_humaneval_n10_XXX
        token = tail[0]
        if "steered" in token:
            # Legacy: <slug>_humaneval_n10_steered
            dataset = "none"
            steered = True
        elif token == "baseline":
            dataset = "none"
            steered = False
        else:
            # Could be a bare dataset tag with no explicit setting
            dataset = token
            steered = False
    else:
        # Pattern: <slug>_humaneval_n10_<dataset>_<setting...>
        dataset = tail[0]
        # Decide steered based on any token containing "steered"
        if any("steered" in t for t in tail[1:]):
            steered = True

    return slug, n, steered, dataset


def main():
    results_dir = "results_100"
    os.makedirs(results_dir, exist_ok=True)

    # Collect ONLY base generation files:
    #   - end with .jsonl
    #   - contain "_humaneval_"
    #   - do NOT contain "results" in the filename
    all_jsonl = sorted(glob.glob(os.path.join(results_dir, "*.jsonl")))
    jsonl_files = []
    for path in all_jsonl:
        fname = os.path.basename(path)
        if "results" in fname:
            # Skip evaluation artifacts like *_results.jsonl, etc.
            continue
        if "_humaneval_" not in fname:
            # Skip non-HumanEval files (e.g., other experiments)
            continue
        jsonl_files.append(path)

    if not jsonl_files:
        print("[WARN] No HumanEval JSONL files found in results/. Did you run gen.py?")
        return

    print(f"[INFO] Found {len(jsonl_files)} HumanEval generation files to evaluate.")

    rows = []

    for path in jsonl_files:
        slug, n, steered, dataset = parse_filename(path)
        print(
            f"[INFO] Evaluating {path} "
            f"(model={slug}, n={n}, steered={steered}, dataset={dataset})"
        )

        # Compute pass@1, pass@5, pass@10
        metrics = evaluate_functional_correctness(
            sample_file=path,
            k=[1, 5, 10],
            n_workers=8,
            timeout=5.0,
        )

        pass1 = metrics.get("pass@1")
        pass5 = metrics.get("pass@5")
        pass10 = metrics.get("pass@10")
        n_samples = metrics.get("n_samples") or metrics.get("n_total")

        row = {
            "file": os.path.basename(path),
            "model": slug,
            "n": n,
            "steered": steered,
            "dataset": dataset,
            "pass@1": pass1,
            "pass@5": pass5,
            "pass@10": pass10,
            "n_samples": n_samples,
        }
        rows.append(row)

    if not rows:
        print("[WARN] No rows collected; nothing to write.")
        return

    df = pd.DataFrame(rows)

    # Timestamp for this evaluation run
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1) Combined summary for all datasets
    combined_csv = os.path.join(results_dir, f"humaneval_summary_all_{ts}.csv")
    df.to_csv(combined_csv, index=False)
    print(f"[INFO] Wrote combined summary to {combined_csv}")

    # 2) Per-dataset summaries under results/<dataset>/
    for dataset, df_ds in df.groupby("dataset"):
        ds = dataset or "none"
        ds_dir = os.path.join(results_dir, ds)
        os.makedirs(ds_dir, exist_ok=True)

        out_csv = os.path.join(ds_dir, f"humaneval_summary_{ds}_{ts}.csv")
        df_ds.to_csv(out_csv, index=False)
        print(f"[INFO] Wrote dataset summary for '{ds}' to {out_csv}")


if __name__ == "__main__":
    main()

