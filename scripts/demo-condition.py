#!/usr/bin/env python3
import json
import os
from itertools import chain
from typing import Dict, List

from activation_steering.config import GlobalConfig
from rich.console import Console
GlobalConfig.console = Console(markup=False)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringDataset, SteeringVector

# ------------------- Config -------------------
DATA_PATH = "/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering-orig/docs/demo-data/bugtype_dataset_small.json"  # <-- path to the dataset we built (with 'train' split)
MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"

# Llama-3-8B has 32 layers; you can widen/narrow this search window as needed
COND_LAYER_RANGE = (16, 31)        # search later layers by default
COND_MAX_LAYERS_TO_COMBINE = 1
COND_THRESHOLD_RANGE = (0.0, 0.06)  # cosine thresholds to sweep
COND_THRESHOLD_STEP = 0.0005

SAVE_DIR = "condition_vectors-qwen2"      # where to save .svec and analysis JSON

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
# ------------------------------------------------

os.makedirs(SAVE_DIR, exist_ok=True)

# 1) Load model/tokenizer
print(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype=torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# 2) Load data (expects 'train' with fields 'base' + bug types)
print(f"Loading dataset: {DATA_PATH}")
with open(DATA_PATH, "r", encoding="utf-8") as f:
    dataset = json.load(f)

train_rows: List[Dict[str, str]] = dataset.get("train", [])
if not train_rows:
    raise RuntimeError("No 'train' examples found in dataset JSON.")

# 3) Collect strings per condition (base + each bug type)
conditions = ["base"] + BUGTYPE_KEYS
data: Dict[str, List[str]] = {c: [] for c in conditions}

for ex in train_rows:
    # "base" is a single string; bug-type keys are optional per row (mosaic rows may omit some)
    if "base" in ex and ex["base"]:
        data["base"].append(ex["base"])
    for bt in BUGTYPE_KEYS:
        s = ex.get(bt)
        if s:
            data[bt].append(s)

# Basic stats
for c in conditions:
    print(f"{c:>32}: {len(data[c]):6d} examples")

# 4) Train a condition vector for each target vs pooled others
for target in conditions:
    # Build positives (target) and negatives (all other conditions pooled)
    positives = data[target]
    if not positives:
        print(f"[skip] No data for target condition: {target}")
        continue

    negatives = list(chain.from_iterable(data[c] for c in conditions if c != target))
    if not negatives:
        print(f"[skip] No negatives available when contrasting: {target}")
        continue

    # Pair construction:
    # We contrast target against each *other* category by concatenating pairs of equal length.
    # This mirrors your original approach (balances counts per other category).
    positive_instructions: List[str] = []
    negative_instructions: List[str] = []
    for other in conditions:
        if other == target:
            continue
        pos_block = positives                           # reuse full target set
        neg_block = data[other]                         # versus this specific other
        # Balance by truncating to the shorter side to avoid heavy duplication
        n = min(len(pos_block), len(neg_block))
        if n == 0:
            continue
        positive_instructions.extend(pos_block[:n])
        negative_instructions.extend(neg_block[:n])

    # Safety check
    n_pairs = min(len(positive_instructions), len(negative_instructions))
    if n_pairs == 0:
        print(f"[skip] Not enough pairs for target={target}")
        continue

    print(f"\n=== Training condition vector for: {target} ===")
    print(f"  pairs: {n_pairs} (pos {len(positive_instructions)} / neg {len(negative_instructions)})")

    # 5) Create the SteeringDataset
    condition_dataset = SteeringDataset(
        tokenizer=tokenizer,
        examples=list(zip(positive_instructions[:n_pairs], negative_instructions[:n_pairs])),
        suffixes=None,
        disable_suffixes=True
    )

    # 6) Train the condition vector (CAST-style condition)
    condition_vector = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=condition_dataset,
        method="pca_pairwise",          # recommended
        accumulate_last_x_tokens="all"  # match IBM quickstart behavior
    )

    # 7) Save the vector
    vec_stem = os.path.join(SAVE_DIR, f"{target}_condition_vector")
    condition_vector.save(vec_stem)
    print(f"  saved: {vec_stem}.svec")

    # 8) Find a good condition point (layer/threshold/direction)
    print("  searching best condition point (layer/threshold)...")
    mal = MalleableModel(model=model, tokenizer=tokenizer)
    best_layer, best_threshold, best_direction, analysis = mal.find_best_condition_point(
        positive_strings=positive_instructions[:n_pairs],
        negative_strings=negative_instructions[:n_pairs],
        condition_vector=condition_vector,
        layer_range=COND_LAYER_RANGE,
        max_layers_to_combine=COND_MAX_LAYERS_TO_COMBINE,
        threshold_range=COND_THRESHOLD_RANGE,
        threshold_step=COND_THRESHOLD_STEP,
        save_analysis=True,
        file_path=os.path.join(SAVE_DIR, f"optimal_condition_point_{target}.json"),
    )

    print(f"  best_layer={best_layer}, best_threshold={best_threshold:.5f}, direction={best_direction}")
    print(f"  analysis saved → {os.path.join(SAVE_DIR, f'optimal_condition_point_{target}.json')}")

print("\nDone. Saved condition vectors and per-condition analyses under:", SAVE_DIR)

