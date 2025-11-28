#!/usr/bin/env python3
import argparse
import json
import csv
import os
from collections import defaultdict


def parse_args():
    ap = argparse.ArgumentParser(
        description="Summarize A/B APR MCQ runs (baseline/steered) from metrics_ab_apr.jsonl."
    )
    ap.add_argument(
        "--metrics_path",
        default="mcq_cache/metrics_ab_apr.jsonl",
        help="Path to metrics JSONL file written by ab_apr_eval.py.",
    )
    ap.add_argument(
        "--output_flat",
        default="mcq_cache/metrics_flat.csv",
        help="Output CSV with one row per run (baseline or steered).",
    )
    ap.add_argument(
        "--output_pairs",
        default="mcq_cache/metrics_pairs.csv",
        help=(
            "Output CSV with baseline/steered paired per "
            "(dataset, model, fewshot_k, layers, strength, seed)."
        ),
    )
    return ap.parse_args()


def load_metrics(path):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Metrics file not found: {path}. "
            "Make sure ab_apr_eval.py has been run and METRICS_PATH is correct."
        )
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                records.append(json.loads(ln))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line in metrics file: {e}")
    return records


def write_flat_csv(records, out_path):
    if not records:
        print("[INFO] No records to write to flat CSV.")
        return

    # Preferred column ordering
    base_order = [
        "timestamp",
        "run_kind",
        "dataset_token",
        "source_dataset",
        "mcq_questions",
        "model_id",
        "mode",
        "fewshot_k",
        "layers",
        "strength",
        "seed",
        "accuracy",
        "invalid_rate",
        "correct",
        "invalid",
        "total",
        "cache_path",
        "vector_path",
    ]

    all_keys = set()
    for r in records:
        all_keys.update(r.keys())

    fieldnames = [k for k in base_order if k in all_keys] + sorted(
        all_keys - set(base_order)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)

    print(f"[INFO] Wrote flat metrics CSV: {out_path}")


def write_pairwise_csv(records, out_path):
    """
    Build a paired view:

    Key: (dataset_token, source_dataset, mcq_questions, model_id,
          fewshot_k, layers, strength, seed)

    For each key, we try to find both a baseline and a steered run.
    """
    grouped = defaultdict(lambda: {"baseline": None, "steered": None})

    for r in records:
        run_kind = r.get("run_kind", "")
        if run_kind not in ("baseline", "steered"):
            continue

        key = (
            r.get("dataset_token", ""),
            r.get("source_dataset", ""),
            r.get("mcq_questions", ""),
            r.get("model_id", ""),
            r.get("fewshot_k", None),
            r.get("layers", ""),
            r.get("strength", None),
            r.get("seed", None),
        )
        grouped[key][run_kind] = r

    # Define columns for paired CSV
    fieldnames = [
        "dataset_token",
        "source_dataset",
        "mcq_questions",
        "model_id",
        "fewshot_k",
        "layers",
        "strength",
        "seed",
        # baseline metrics
        "baseline_accuracy",
        "baseline_invalid_rate",
        "baseline_correct",
        "baseline_invalid",
        "baseline_total",
        "baseline_cache_path",
        # steered metrics
        "steered_accuracy",
        "steered_invalid_rate",
        "steered_correct",
        "steered_invalid",
        "steered_total",
        "steered_cache_path",
        "steered_vector_path",
        # deltas
        "delta_accuracy",
        "delta_invalid_rate",
    ]

    rows = []
    for key, runs in grouped.items():
        (dataset_token, source_ds, mcq_path, model_id, k, layers, strength, seed) = key
        base = runs.get("baseline")
        steered = runs.get("steered")

        # It is okay if only one side exists; the other fields will be blank.
        row = {
            "dataset_token": dataset_token,
            "source_dataset": source_ds,
            "mcq_questions": mcq_path,
            "model_id": model_id,
            "fewshot_k": k,
            "layers": layers,
            "strength": strength,
            "seed": seed,
        }

        # Baseline part
        if base is not None:
            row["baseline_accuracy"] = base.get("accuracy")
            row["baseline_invalid_rate"] = base.get("invalid_rate")
            row["baseline_correct"] = base.get("correct")
            row["baseline_invalid"] = base.get("invalid")
            row["baseline_total"] = base.get("total")
            row["baseline_cache_path"] = base.get("cache_path")
        else:
            row["baseline_accuracy"] = ""
            row["baseline_invalid_rate"] = ""
            row["baseline_correct"] = ""
            row["baseline_invalid"] = ""
            row["baseline_total"] = ""
            row["baseline_cache_path"] = ""

        # Steered part
        if steered is not None:
            row["steered_accuracy"] = steered.get("accuracy")
            row["steered_invalid_rate"] = steered.get("invalid_rate")
            row["steered_correct"] = steered.get("correct")
            row["steered_invalid"] = steered.get("invalid")
            row["steered_total"] = steered.get("total")
            row["steered_cache_path"] = steered.get("cache_path")
            row["steered_vector_path"] = steered.get("vector_path")
        else:
            row["steered_accuracy"] = ""
            row["steered_invalid_rate"] = ""
            row["steered_correct"] = ""
            row["steered_invalid"] = ""
            row["steered_total"] = ""
            row["steered_cache_path"] = ""
            row["steered_vector_path"] = ""

        # Deltas (only if both exist and values are numeric)
        try:
            b_acc = float(row["baseline_accuracy"])
            s_acc = float(row["steered_accuracy"])
            row["delta_accuracy"] = s_acc - b_acc
        except (TypeError, ValueError):
            row["delta_accuracy"] = ""

        try:
            b_inv = float(row["baseline_invalid_rate"])
            s_inv = float(row["steered_invalid_rate"])
            row["delta_invalid_rate"] = s_inv - b_inv
        except (TypeError, ValueError):
            row["delta_invalid_rate"] = ""

        rows.append(row)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote paired metrics CSV: {out_path}")


def main():
    args = parse_args()
    records = load_metrics(args.metrics_path)
    if not records:
        print("[INFO] No records found in metrics file; nothing to summarize.")
        return

    write_flat_csv(records, args.output_flat)
    write_pairwise_csv(records, args.output_pairs)


if __name__ == "__main__":
    main()

