#!/usr/bin/env python
"""
compute_tssb_k0_metrics.py

Parse ONLY k=0 TSSB MCQ results (baseline + steered) and write summary CSVs.

Assumes you already ran process_tssb_metrics.py so you have:
  mcq_cache/tssb/tssb_baseline_index.csv
  mcq_cache/tssb/tssb_steered_index.csv

Assumes file layout like:
  mcq_cache/
    tssb/
      baseline/
        baseline_tssb_*.json
      steered/
        steered_tssb_mcq_k0_seed*_*.json

Outputs:
  mcq_cache/tssb/tssb_k0_metrics.csv
  mcq_cache/tssb/tssb_k0_best_configs.csv
"""

import csv
import json
import os
from typing import Any, Dict, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT = os.path.dirname(ROOT)                        # repo root

MCQ_ROOT = os.path.join(ROOT, "mcq_cache")
TSSB_ROOT = os.path.join(MCQ_ROOT, "tssb")

BASELINE_INDEX = os.path.join(TSSB_ROOT, "tssb_baseline_index.csv")
STEERED_INDEX = os.path.join(TSSB_ROOT, "tssb_steered_index.csv")

# ----------------------------------------------------------------------
# Helper: robust-ish metric extractor from result JSON
# ----------------------------------------------------------------------

def extract_metrics(obj: Any) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[float]]:
    """
    Try to extract (num_correct, num_total, num_invalid, accuracy) from a
    result JSON object.

    IMPORTANT:
    - You may need to tweak this depending on how ab_apr_eval.py logs things.
    - This tries several common patterns.

    Returns (num_correct, num_total, num_invalid, accuracy).
    Any of them can be None if not found.
    """

    # Case 1: top-level dict with standard metrics
    if isinstance(obj, dict):
        # Direct aggregate fields
        for key in ["accuracy", "acc"]:
            if key in obj and isinstance(obj[key], (int, float)):
                acc = float(obj[key])
                # Try to infer counts if present
                n_correct = obj.get("num_correct") or obj.get("correct") or obj.get("n_correct")
                n_total = obj.get("num_total") or obj.get("total") or obj.get("n_total")
                n_invalid = obj.get("num_invalid") or obj.get("invalid") or obj.get("n_invalid")

                # Cast to ints where possible
                n_correct = int(n_correct) if n_correct is not None else None
                n_total = int(n_total) if n_total is not None else None
                n_invalid = int(n_invalid) if n_invalid is not None else None
                return n_correct, n_total, n_invalid, acc

        # Nested summary dict (e.g., obj["summary"]["accuracy"])
        for summary_key in ["summary", "metrics"]:
            if summary_key in obj and isinstance(obj[summary_key], dict):
                s = obj[summary_key]
                for key in ["accuracy", "acc"]:
                    if key in s:
                        acc = float(s[key])
                        n_correct = s.get("num_correct") or s.get("correct") or s.get("n_correct")
                        n_total = s.get("num_total") or s.get("total") or s.get("n_total")
                        n_invalid = s.get("num_invalid") or s.get("invalid") or s.get("n_invalid")
                        n_correct = int(n_correct) if n_correct is not None else None
                        n_total = int(n_total) if n_total is not None else None
                        n_invalid = int(n_invalid) if n_invalid is not None else None
                        return n_correct, n_total, n_invalid, acc

    # Case 2: list of per-example records
    if isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], dict):
        records = obj
        n_total = len(records)
        n_correct = 0
        n_invalid = 0

        # Heuristics for per-example fields
        for rec in records:
            # invalid flag
            if "is_invalid" in rec:
                if rec["is_invalid"]:
                    n_invalid += 1

            # correctness
            if "is_correct" in rec:
                if rec["is_correct"]:
                    n_correct += 1
            elif "correct" in rec and isinstance(rec["correct"], bool):
                if rec["correct"]:
                    n_correct += 1
            elif "gold" in rec and "pred" in rec:
                if rec["pred"] == rec["gold"]:
                    n_correct += 1

        if n_total > 0:
            acc = n_correct / n_total
        else:
            acc = None
        return n_correct, n_total, n_invalid if n_invalid > 0 else None, acc

    # Fallback: nothing recognized
    return None, None, None, None


# ----------------------------------------------------------------------
# Load index CSVs
# ----------------------------------------------------------------------

def load_index(path: str):
    rows = []
    if not os.path.isfile(path):
        print(f"[ERROR] Index file not found: {path}")
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# ----------------------------------------------------------------------
# Main metric computation for k=0 only
# ----------------------------------------------------------------------

def main():
    print(f"[INFO] ROOT = {ROOT}")
    print(f"[INFO] Using baseline index: {BASELINE_INDEX}")
    print(f"[INFO] Using steered index : {STEERED_INDEX}")

    baseline_rows = load_index(BASELINE_INDEX)
    steered_rows = load_index(STEERED_INDEX)

    # Filter to k=0 only
    baseline_k0 = [r for r in baseline_rows if int(r["fewshot_k"]) == 0]
    steered_k0 = [r for r in steered_rows if int(r["fewshot_k"]) == 0]

    print(f"[INFO] Baseline k=0 entries: {len(baseline_k0)}")
    print(f"[INFO] Steered  k=0 entries: {len(steered_k0)}")

    all_rows = []

    # --- Process baselines ---
    for r in baseline_k0:
        model_slug = r["model_slug"]
        k = int(r["fewshot_k"])
        seed = int(r["seed"])
        rel_path = r["baseline_path"]  # e.g. tssb/baseline/...
        abs_path = os.path.join(MCQ_ROOT, rel_path)

        try:
            with open(abs_path, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load baseline {abs_path}: {e}")
            n_correct = n_total = n_invalid = acc = None
        else:
            n_correct, n_total, n_invalid, acc = extract_metrics(obj)

        all_rows.append(
            {
                "dataset": "tssb",
                "mode": "baseline",
                "model_slug": model_slug,
                "fewshot_k": k,
                "seed": seed,
                "layers": "",
                "strength_tag": "",
                "num_correct": n_correct,
                "num_total": n_total,
                "num_invalid": n_invalid,
                "accuracy": acc,
                "path": rel_path,
            }
        )

    # --- Process steered ---
    for r in steered_k0:
        model_slug = r["model_slug"]
        k = int(r["fewshot_k"])
        seed = int(r["seed"])
        layers = r["layers"]          # e.g. "band-0.15-6"
        strength_tag = r["strength_tag"]  # e.g. "1p0"
        rel_path = r["steered_path"]
        abs_path = os.path.join(MCQ_ROOT, rel_path)

        try:
            with open(abs_path, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load steered {abs_path}: {e}")
            n_correct = n_total = n_invalid = acc = None
        else:
            n_correct, n_total, n_invalid, acc = extract_metrics(obj)

        all_rows.append(
            {
                "dataset": "tssb",
                "mode": "steered",
                "model_slug": model_slug,
                "fewshot_k": k,
                "seed": seed,
                "layers": layers,
                "strength_tag": strength_tag,
                "num_correct": n_correct,
                "num_total": n_total,
                "num_invalid": n_invalid,
                "accuracy": acc,
                "path": rel_path,
            }
        )

    # Write full metrics table
    metrics_path = os.path.join(TSSB_ROOT, "tssb_k0_metrics.csv")
    fieldnames = [
        "dataset",
        "mode",
        "model_slug",
        "fewshot_k",
        "seed",
        "layers",
        "strength_tag",
        "num_correct",
        "num_total",
        "num_invalid",
        "accuracy",
        "path",
    ]
    with open(metrics_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"[INFO] Wrote k=0 metrics table: {metrics_path} (rows={len(all_rows)})")

    # ------------------------------------------------------------------
    # Build "best config per model" table (steered vs baseline)
    # ------------------------------------------------------------------

    # Index baseline accuracy per model
    baseline_acc_by_model: Dict[str, float] = {}
    for row in all_rows:
        if row["mode"] != "baseline":
            continue
        if row["accuracy"] is None:
            continue
        baseline_acc_by_model[row["model_slug"]] = float(row["accuracy"])

    # For each model, find best steered config
    best_rows = []
    for model_slug, base_acc in baseline_acc_by_model.items():
        best: Optional[Dict[str, Any]] = None
        for row in all_rows:
            if row["mode"] != "steered":
                continue
            if row["model_slug"] != model_slug:
                continue
            if row["accuracy"] is None:
                continue
            acc = float(row["accuracy"])
            if best is None or acc > float(best["accuracy"]):
                best = row

        if best is None:
            # no steered configs for this model (shouldn't happen for k=0)
            continue

        best_acc = float(best["accuracy"])
        delta = best_acc - base_acc

        best_rows.append(
            {
                "dataset": "tssb",
                "model_slug": model_slug,
                "fewshot_k": best["fewshot_k"],
                "baseline_acc": base_acc,
                "best_steered_acc": best_acc,
                "delta_acc": delta,
                "best_layers": best["layers"],
                "best_strength_tag": best["strength_tag"],
                "best_path": best["path"],
            }
        )

    best_path = os.path.join(TSSB_ROOT, "tssb_k0_best_configs.csv")
    fieldnames_best = [
        "dataset",
        "model_slug",
        "fewshot_k",
        "baseline_acc",
        "best_steered_acc",
        "delta_acc",
        "best_layers",
        "best_strength_tag",
        "best_path",
    ]
    with open(best_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_best)
        writer.writeheader()
        writer.writerows(best_rows)

    print(f"[INFO] Wrote best-configs table: {best_path} (rows={len(best_rows)})")


if __name__ == "__main__":
    main()

