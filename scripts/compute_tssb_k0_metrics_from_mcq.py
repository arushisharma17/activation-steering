#!/usr/bin/env python
"""
compute_tssb_k0_metrics_from_mcq.py

Compute ONLY k=0 TSSB MCQ metrics (baseline + steered) by
aligning raw_outputs with MCQ questions and gold labels.

Requires:
  mcq_cache/tssb/tssb_baseline_index.csv
  mcq_cache/tssb/tssb_steered_index.csv
  mcq_cache/tssb/mcq/tssb_mcq_k0_seed<SEED>.json

Outputs:
  mcq_cache/tssb/tssb_k0_metrics.csv
  mcq_cache/tssb/tssb_k0_best_configs.csv
"""

import csv
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

ROOT = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT = os.path.dirname(ROOT)                        # repo root

MCQ_ROOT = os.path.join(ROOT, "mcq_cache")
TSSB_ROOT = os.path.join(MCQ_ROOT, "tssb")

BASELINE_INDEX = os.path.join(TSSB_ROOT, "tssb_baseline_index.csv")
STEERED_INDEX = os.path.join(TSSB_ROOT, "tssb_steered_index.csv")


# ---------------------------------------------------------------------
# Utilities: load CSV indices
# ---------------------------------------------------------------------

def load_index(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not os.path.isfile(path):
        print(f"[ERROR] Index file not found: {path}")
        return rows
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows


# ---------------------------------------------------------------------
# MCQ + outputs -> metrics
# ---------------------------------------------------------------------
def load_mcq(dataset: str, k: int, seed: int) -> List[Dict[str, Any]]:
    """
    Load MCQ questions for (dataset, k, seed).

    For TSSB, expect something like either:
      [ {...}, {...}, ... ]                         # direct list
    or
      { "eval": [ {...}, ... ], ... }               # dict containing a list
    or
      { "<whatever>": [ {...}, ... ], ... }         # any key whose value is a list

    We just pick the *first* list-valued field if the top-level object is a dict.
    """
    if dataset != "tssb":
        raise ValueError(f"Only dataset='tssb' is supported (got {dataset})")

    rel = os.path.join("tssb", "mcq", f"tssb_mcq_k{k}_seed{seed}.json")
    path = os.path.join(MCQ_ROOT, rel)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"MCQ file not found: {path}")

    with open(path, "r") as f:
        data = json.load(f)

    # Case 1: already a list
    if isinstance(data, list):
        return data

    # Case 2: dict containing one or more list-valued fields
    if isinstance(data, dict):
        # Prefer some common keys if present
        for key in ["eval", "questions", "data", "items"]:
            if key in data and isinstance(data[key], list):
                return data[key]

        # Otherwise, just take the first list-valued entry
        for key, value in data.items():
            if isinstance(value, list):
                return value

        raise ValueError(
            f"MCQ file {path} is a dict but has no list-valued fields; "
            f"keys={list(data.keys())}"
        )

    raise ValueError(
        f"MCQ file {path} must be or contain a list of questions, "
        f"got type {type(data)}"
    )


def parse_pred_label(raw: str) -> Optional[str]:
    """
    Given a raw output string (e.g., " B", " A) The", " Answer: B", " B</s>"),
    try to extract a prediction label: 'A' or 'B'.

    Heuristics:
      1) Look for "Answer: X" pattern
      2) Otherwise, find first 'A' or 'B' character in the string
    """
    if raw is None:
        return None
    s = str(raw).strip()

    # Pattern: "Answer: B"
    m = re.search(r"Answer:\s*([AB])", s)
    if m:
        return m.group(1)

    # Otherwise, scan characters left-to-right and pick first A/B
    for ch in s:
        if ch in ("A", "B"):
            return ch

    return None


def compute_metrics_from_mcq_and_outputs(
    mcq: List[Dict[str, Any]],
    outputs: List[str],
) -> Tuple[int, int, int, float]:
    """
    Given MCQ questions + raw_outputs (aligned by index),
    compute (num_correct, num_total, num_invalid, accuracy).

    num_invalid counts entries where we couldn't parse a prediction.
    """
    n = min(len(mcq), len(outputs))
    if n == 0:
        return 0, 0, 0, 0.0

    num_correct = 0
    num_invalid = 0

    for i in range(n):
        q = mcq[i]
        gold = q.get("gold", None)
        pred = parse_pred_label(outputs[i])

        if pred is None or gold is None:
            num_invalid += 1
            continue

        # normalize just in case
        gold = str(gold).strip()
        pred = str(pred).strip()
        if gold not in ("A", "B") or pred not in ("A", "B"):
            num_invalid += 1
            continue

        if pred == gold:
            num_correct += 1

    num_total = n
    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return num_correct, num_total, num_invalid, accuracy


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

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

    all_rows: List[Dict[str, Any]] = []

    # We'll assume dataset = 'tssb' for everything in these index files.
    dataset = "tssb"

    # Pre-load MCQ per seed (in case you ever use multiple seeds)
    mcq_cache_by_seed: Dict[int, List[Dict[str, Any]]] = {}

    def get_mcq_for(seed: int) -> List[Dict[str, Any]]:
        if seed not in mcq_cache_by_seed:
            mcq_cache_by_seed[seed] = load_mcq(dataset, k=0, seed=seed)
            print(f"[INFO] Loaded MCQ for seed={seed}, n={len(mcq_cache_by_seed[seed])}")
        return mcq_cache_by_seed[seed]

    # -------- Baseline --------
    for r in baseline_k0:
        model_slug = r["model_slug"]
        seed = int(r["seed"])
        rel_path = r["baseline_path"]  # e.g. "tssb/baseline/..."
        abs_path = os.path.join(MCQ_ROOT, rel_path)

        # Load MCQ and raw_outputs
        mcq = get_mcq_for(seed)

        try:
            with open(abs_path, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load baseline {abs_path}: {e}")
            n_correct = n_total = n_invalid = 0
            acc = 0.0
        else:
            outputs = obj.get("raw_outputs", None)
            if not isinstance(outputs, list):
                print(f"[WARN] baseline {abs_path} missing 'raw_outputs' list; treating as empty.")
                outputs = []
            n_correct, n_total, n_invalid, acc = compute_metrics_from_mcq_and_outputs(mcq, outputs)

        all_rows.append(
            {
                "dataset": dataset,
                "mode": "baseline",
                "model_slug": model_slug,
                "fewshot_k": 0,
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

    # -------- Steered --------
    for r in steered_k0:
        model_slug = r["model_slug"]
        seed = int(r["seed"])
        layers = r["layers"]
        strength_tag = r["strength_tag"]
        rel_path = r["steered_path"]
        abs_path = os.path.join(MCQ_ROOT, rel_path)

        mcq = get_mcq_for(seed)

        try:
            with open(abs_path, "r") as f:
                obj = json.load(f)
        except Exception as e:
            print(f"[WARN] Failed to load steered {abs_path}: {e}")
            n_correct = n_total = n_invalid = 0
            acc = 0.0
        else:
            outputs = obj.get("raw_outputs", None)
            if not isinstance(outputs, list):
                print(f"[WARN] steered {abs_path} missing 'raw_outputs' list; treating as empty.")
                outputs = []
            n_correct, n_total, n_invalid, acc = compute_metrics_from_mcq_and_outputs(mcq, outputs)

        all_rows.append(
            {
                "dataset": dataset,
                "mode": "steered",
                "model_slug": model_slug,
                "fewshot_k": 0,
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

    # -------- Write full metrics table --------
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

    # -------- Best config per model --------
    baseline_acc_by_model: Dict[str, float] = {}
    for row in all_rows:
        if row["mode"] != "baseline":
            continue
        baseline_acc_by_model[row["model_slug"]] = float(row["accuracy"])

    best_rows: List[Dict[str, Any]] = []
    for model_slug, base_acc in baseline_acc_by_model.items():
        best: Optional[Dict[str, Any]] = None
        for row in all_rows:
            if row["mode"] != "steered":
                continue
            if row["model_slug"] != model_slug:
                continue
            acc = float(row["accuracy"])
            if best is None or acc > float(best["accuracy"]):
                best = row

        if best is None:
            continue

        best_acc = float(best["accuracy"])
        delta = best_acc - base_acc

        best_rows.append(
            {
                "dataset": dataset,
                "model_slug": model_slug,
                "fewshot_k": 0,
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

