#!/usr/bin/env python3
import os, json
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from activation_steering import MalleableModel, SteeringVector
from activation_steering.config import GlobalConfig
from rich.console import Console

# --------- Safety: avoid Rich trying to parse [INST] tags as markup ----------
GlobalConfig.console = Console(markup=False)

# ===================== Config =====================
MODEL_ID = "NousResearch/Hermes-2-Pro-Llama-3-8B"

# Directory where you saved the condition vectors + optimal points
COND_DIR = "condition_vectors"

# Behavior vector: should be a *repair* behavior direction, e.g., (fixed - buggy).
# Use your actual path stem (without .svec). Example:
BEHAVIOR_VEC_STEM = "apr_fix_behavior"  # e.g., "apr_fix_behavior" -> "apr_fix_behavior.svec"

# These are the 9 “conditions” we expect (will skip any missing)
BUGTYPE_KEYS = [
    "base",
    "MORE_SPECIFIC_IF",
    "ADD_METHOD_CALL",
    "ADD_FUNCTION_AROUND_EXPRESSION",
    "SAME_FUNCTION_MORE_ARGS",
    "SAME_FUNCTION_LESS_ARGS",
    "CHANGE_BINARY_OPERATOR",
    "CHANGE_COMPARISON_OPERATOR",
    "SINGLE_TOKEN",
]
# ==================================================

def load_cond(name: str) -> Tuple[SteeringVector, List[int], float, str]:
    """
    Load a condition vector and its best (layer, threshold, direction) from JSON.
    Returns: (vector, [layer], threshold, comparator_str)
    comparator_str is 'larger' if direction=='positive' else 'smaller'.
    """
    vec_path = os.path.join(COND_DIR, f"{name}_condition_vector.svec")
    gate_path = os.path.join(COND_DIR, f"optimal_condition_point_{name}.json")
    if not os.path.isfile(vec_path) or not os.path.isfile(gate_path):
        raise FileNotFoundError(f"Missing files for condition '{name}': {vec_path} or {gate_path}")

    vec = SteeringVector.load(vec_path)
    gate = json.load(open(gate_path, "r"))
    best_layer = gate.get("best_layer")
    best_threshold = float(gate.get("best_threshold", 0.0))
    direction = gate.get("best_direction", "positive")
    comparator = "larger" if direction == "positive" else "smaller"
    return vec, [int(best_layer)], best_threshold, comparator

def main():
    # 1) Load model + tokenizer
    print(f"Loading model: {MODEL_ID}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    # 2) Load behavior (APR) vector
    behavior_vec_path = f"{BEHAVIOR_VEC_STEM}.svec"
    if not os.path.isfile(behavior_vec_path):
        raise FileNotFoundError(
            f"Behavior vector not found: {behavior_vec_path}\n"
            f"Set BEHAVIOR_VEC_STEM to your repair vector (e.g., 'apr_fix_behavior')."
        )
    behavior_vec = SteeringVector.load(BEHAVIOR_VEC_STEM)

    # 3) Collect available conditions
    condition_vectors = []
    condition_layer_ids = []
    condition_thresholds = []
    condition_comparators = []
    cond_index_map: Dict[str, int] = {}

    print("\nLoading condition vectors & gates...")
    for name in BUGTYPE_KEYS:
        try:
            vec, layers, thr, comp = load_cond(name)
        except FileNotFoundError:
            print(f"  - [skip] {name}: missing files in {COND_DIR}/")
            continue
        cond_index_map[name] = len(condition_vectors) + 1  # 1-based index for multisteer rules (C1, C2, ...)
        condition_vectors.append(vec)
        condition_layer_ids.append(layers)
        condition_thresholds.append(thr)
        condition_comparators.append(comp)
        print(f"  - {name:35s} layer={layers[0]:2d} thr={thr:.5f} dir={'>' if comp=='larger' else '<'}0")

    if not condition_vectors:
        raise RuntimeError("No condition vectors found. Make sure COND_DIR has *.svec and optimal JSONs.")

    # 4) Build a simple rule: if (any condition) then apply behavior B1
    #    Map to C1..Ck according to the order we appended above.
    all_C = [f"C{i}" for i in range(1, len(condition_vectors) + 1)]
    rule_any = f"if {' or '.join(all_C)} then B1"

    # 5) Wrap the model
    mal = MalleableModel(model=model, tokenizer=tokenizer)

    # Choose some behavior layers (late layers tend to work well; tweak as needed)
    behavior_layers = [15, 16, 17, 18, 19, 20, 21, 22, 23]
    behavior_strengths = [1.2]  # try 0.5 / 1.0 / 1.5 sweeps

    print("\nApplying multisteer with rule:")
    print("  ", rule_any)

    mal.multisteer(
        behavior_vectors=[behavior_vec],
        behavior_layer_ids=[behavior_layers],
        behavior_vector_strengths=behavior_strengths,
        condition_vectors=condition_vectors,
        condition_layer_ids=condition_layer_ids,
        condition_vector_thresholds=condition_thresholds,
        condition_comparator_threshold_is=condition_comparators,
        rules=[rule_any],
    )

    # 6) Quick smoke test prompts (swap in your real eval prompts)
    test_prompts = [
        # a generic “base”-ish instruction
        "Fix the given buggy code. Return only the corrected code, with no extra commentary.\n\nBuggy:\nvalue = urllib . quote ( value )\n\nAnswer:",
        # a SINGLE_TOKEN-like snippet
        "Fix the given buggy code. Return only the corrected code, with no extra commentary.\n\nBuggy:\ndef copy_op_with_new_args ( self , args ) : return type ( self ) ( args [ 0 ] , axes = self . axes )\n\nAnswer:",
        # a CHANGE_BINARY_OPERATOR-like snippet
        "Fix the given buggy code. Return only the corrected code, with no extra commentary.\n\nBuggy:\nif a + b == c:\n    return True\n\nAnswer:",
    ]

    outputs = mal.respond_batch_sequential(prompts=test_prompts)
    print("\n=== Sample outputs ===")
    for i, out in enumerate(outputs):
        print(f"\nPrompt {i}:\n{test_prompts[i]}\n---\n{out}")

if __name__ == "__main__":
    main()

