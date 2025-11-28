#!/usr/bin/env python
"""
process_tssb_metrics.py

TSSB-only metrics/index parser.

Run this from the repo root, e.g.:

    cd /lustre/hdd/LAS/jannesar-lab/arushi/activation-steering
    python scripts/process_tssb_metrics.py

Assumes the following structure already exists (after running organize_mcq_cache.sh):

    mcq_cache/
        tssb/
            baseline/
                baseline_tssb_*.json
            steered/
                steered_tssb_mcq_k*_seed*_*.json
            mcq/
                tssb_mcq_k*_seed*.json

Outputs (in mcq_cache/tssb/):

    - tssb_baseline_index.csv
    - tssb_steered_index.csv
    - tssb_steered_coverage_summary.csv
"""

import csv
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.abspath(__file__))  # scripts/
ROOT = os.path.dirname(ROOT)                        # repo root

MCQ_ROOT = os.path.join(ROOT, "mcq_cache")
TSSB_ROOT = os.path.join(MCQ_ROOT, "tssb")
BASELINE_DIR = os.path.join(TSSB_ROOT, "baseline")
STEERED_DIR = os.path.join(TSSB_ROOT, "steered")

os.makedirs(TSSB_ROOT, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Helpers
# ---------------------------------------------------------------------------

# baseline_tssb_codellama-7b-instruct-hf_k0_seed42.json
BASELINE_RE = re.compile(
    r"^baseline_tssb_(?P<model_slug>.+)_k(?P<k>\d+)_seed(?P<seed>\d+)\.json$"
)

# steered_tssb_mcq_k0_seed42_meta-llama-CodeLlama-7b-Instruct-hf_k0_Lband-0.15-6_a1p0_seed42_es0_el5000.json
STEERED_RE = re.compile(
    r"^steered_tssb_mcq_k(?P<k>\d+)_seed(?P<seed>\d+)_(?P<model_id>.+?)_k\d+_L(?P<layers>[^_]+)_a(?P<alpha>[^_]+)_seed(?P<seed2>\d+)_es(?P<es>\d+)_el(?P<el>\d+)\.json$"
)


def normalize_model_slug(model_id: str) -> str:
    """
    Map HF-ish model IDs in filenames to the normalized slugs you use elsewhere.

    Examples of model_id in filenames:
        meta-llama-CodeLlama-7b-Instruct-hf
        Qwen-Qwen2.5-7B-Instruct
        Qwen-Qwen2.5-Coder-7B-Instruct
        Qwen-Qwen2.5-Coder-14B-Instruct

    Adjust this mapping if needed.
    """
    mid = model_id.lower()

    if "codellama-7b-instruct-hf" in mid:
        return "codellama-7b-instruct-hf"
    if "qwen2.5-7b-instruct" in mid or "qwen-7b-instruct" in mid:
        return "qwen2-5-7b-instruct"
    if "qwen2.5-coder-7b-instruct" in mid or "coder-7b-instruct" in mid:
        return "qwen2-5-coder-7b-instruct"
    if "qwen2.5-coder-14b-instruct" in mid or "coder-14b-instruct" in mid:
        return "qwen2-5-coder-14b-instruct"

    # Fallback: just return the raw model_id
    return model_id


# ---------------------------------------------------------------------------
# 2. Index baseline files
# ---------------------------------------------------------------------------

def index_baseline():
    rows = []
    if not os.path.isdir(BASELINE_DIR):
        print(f"[WARN] Baseline dir does not exist: {BASELINE_DIR}")
        return rows

    for fname in sorted(os.listdir(BASELINE_DIR)):
        m = BASELINE_RE.match(fname)
        if not m:
            # skip non-baseline files or weird names
            continue
        model_slug = m.group("model_slug")
        k = int(m.group("k"))
        seed = int(m.group("seed"))
        path = os.path.join("tssb", "baseline", fname)  # relative to mcq_cache

        rows.append(
            {
                "dataset": "tssb",
                "model_slug": model_slug,
                "fewshot_k": k,
                "seed": seed,
                "baseline_path": path,
            }
        )

    out_path = os.path.join(TSSB_ROOT, "tssb_baseline_index.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "model_slug", "fewshot_k", "seed", "baseline_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote baseline index: {out_path} ({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# 3. Index steered files
# ---------------------------------------------------------------------------

def index_steered():
    rows = []
    if not os.path.isdir(STEERED_DIR):
        print(f"[WARN] Steered dir does not exist: {STEERED_DIR}")
        return rows

    for fname in sorted(os.listdir(STEERED_DIR)):
        m = STEERED_RE.match(fname)
        if not m:
            # skip anything that doesn't match our pattern
            continue

        k = int(m.group("k"))
        seed = int(m.group("seed"))
        seed2 = int(m.group("seed2"))
        model_id = m.group("model_id")
        layers = m.group("layers")      # e.g. "band-0.15-6"
        alpha = m.group("alpha")        # e.g. "1p0"
        es = int(m.group("es"))
        el = int(m.group("el"))

        model_slug = normalize_model_slug(model_id)
        path = os.path.join("tssb", "steered", fname)  # relative to mcq_cache

        if seed != seed2:
            print(
                f"[WARN] Seed mismatch in steered file {fname}: "
                f"seed={seed}, seed2={seed2}"
            )

        rows.append(
            {
                "dataset": "tssb",
                "model_id_raw": model_id,
                "model_slug": model_slug,
                "fewshot_k": k,
                "seed": seed,
                "layers": layers,
                "strength_tag": alpha,
                "eval_start": es,
                "eval_limit": el,
                "steered_path": path,
            }
        )

    out_path = os.path.join(TSSB_ROOT, "tssb_steered_index.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model_id_raw",
                "model_slug",
                "fewshot_k",
                "seed",
                "layers",
                "strength_tag",
                "eval_start",
                "eval_limit",
                "steered_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote steered index: {out_path} ({len(rows)} rows)")
    return rows


# ---------------------------------------------------------------------------
# 4. Coverage summary: how complete is TSSB?
# ---------------------------------------------------------------------------

def build_coverage_summary(baseline_rows, steered_rows):
    # What model/k combos do we have baselines for?
    baseline_keys = set(
        (r["model_slug"], r["fewshot_k"]) for r in baseline_rows
    )

    # For each model/k, how many steered configs do we have?
    steered_counts = defaultdict(int)
    steered_config_sets = defaultdict(set)

    for r in steered_rows:
        key = (r["model_slug"], r["fewshot_k"])
        steered_counts[key] += 1
        steered_config_sets[key].add((r["layers"], r["strength_tag"]))

    # Gather all model/k combos we see anywhere
    all_keys = set(baseline_keys) | set(steered_counts.keys())

    # Build summary rows
    summary_rows = []
    for (model_slug, k) in sorted(all_keys, key=lambda x: (x[0], x[1])):
        has_baseline = (model_slug, k) in baseline_keys
        num_steered = steered_counts.get((model_slug, k), 0)
        num_unique_configs = len(steered_config_sets.get((model_slug, k), set()))

        summary_rows.append(
            {
                "dataset": "tssb",
                "model_slug": model_slug,
                "fewshot_k": k,
                "has_baseline": int(has_baseline),
                "num_steered_files": num_steered,
                "num_steered_configs": num_unique_configs,
            }
        )

    out_path = os.path.join(TSSB_ROOT, "tssb_steered_coverage_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset",
                "model_slug",
                "fewshot_k",
                "has_baseline",
                "num_steered_files",
                "num_steered_configs",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[INFO] Wrote coverage summary: {out_path} ({len(summary_rows)} rows)")

    # Also print a quick human-readable summary
    print("\n[SUMMARY] TSSB coverage by model and k:")
    for r in summary_rows:
        print(
            f"  model={r['model_slug']:26s}  k={r['fewshot_k']}  "
            f"baseline={r['has_baseline']}  "
            f"steered_files={r['num_steered_files']}  "
            f"configs={r['num_steered_configs']}"
        )


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------

def main():
    print(f"[INFO] MCQ_ROOT = {MCQ_ROOT}")
    print(f"[INFO] TSSB_ROOT = {TSSB_ROOT}")
    print(f"[INFO] Baseline dir = {BASELINE_DIR}")
    print(f"[INFO] Steered dir  = {STEERED_DIR}")

    baseline_rows = index_baseline()
    steered_rows = index_steered()
    build_coverage_summary(baseline_rows, steered_rows)


if __name__ == "__main__":
    main()

