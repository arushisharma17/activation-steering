#!/usr/bin/env python3
import json
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

HF_cache = ""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0, help="Seed for A/B randomization (0 = nondeterministic)")
    ap.add_argument("--vector_path", type=str, default="refusal_behavior_vector", help="Path to SteeringVector.load")
    ap.add_argument("--strength", type=float, default=2.0, help="Steering vector strength")
    ap.add_argument("--layers", type=str, default="27,28,29,30,31", help="Comma-separated layer ids to steer")
    ap.add_argument("--model_id", type=str, default="meta-llama/CodeLlama-7b-Instruct-hf")
    return ap.parse_args()

# ------------- Few-shot A/B header (fixed answers teach the format) -------------
FEW_SHOT_AB = """\
<s>[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.
Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.

A) dns = dict()
B) dns = dict(default=None, type='list')
Answer: B [/INST]

[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.
Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.

A) parts = env_var.split('=')
B) parts = env_var.split('=', 1)
Answer: B [/INST]

[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.
Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.

A) security_opts = dict(type=list)
B) security_opts = dict(type='list')
Answer: B [/INST]
"""

def build_inst_prompt(a_line: str, b_line: str) -> str:
    """Wrap one randomized A/B example as an [INST] block with an Answer: stub."""
    return (
        FEW_SHOT_AB
        + "[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.\n"
          "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
          f"A) {a_line}\n"
          f"B) {b_line}\n"
          "Answer: [/INST]"
    )

def extract_choice_ab(text: str) -> str:
    """Robustly extract 'A' or 'B' from model output."""
    s = (text or "").strip().upper()
    if s.startswith("A"): return "A"
    if s.startswith("B"): return "B"
    for key in ("ANSWER:", "ANSWER IS", "CORRECT:", "CHOICE:"):
        idx = s.find(key)
        if idx != -1:
            tail = s[idx+len(key):].strip()
            if tail.startswith("A"): return "A"
            if tail.startswith("B"): return "B"
    # Scan whitespace/punct-separated tokens
    for tok in s.replace(")", " ").replace(".", " ").split():
        if tok == "A": return "A"
        if tok == "B": return "B"
    return ""  # invalid/unclear

def main():
    args = parse_args()
    if args.seed:
        random.seed(args.seed)

    # -------- Load model/tokenizer --------
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=HF_cache,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=HF_cache)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # -------- Load steering vector & wrap --------
    vec = SteeringVector.load(args.vector_path)
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    malleable_model = MalleableModel(model=model, tokenizer=tokenizer)
    malleable_model.steer(
        behavior_vector=vec,
        behavior_layer_ids=layers,
        behavior_vector_strength=args.strength,
    )

    # -------- Your pairs: (buggy_before, fixed_after) --------
    pairs = [
        ("dns = dict()", "dns = dict(default=None, type='list')"),
        ("parts = env_var.split('=')", "parts = env_var.split('=', 1)"),
        ("security_opts = dict(type=list)", "security_opts = dict(type='list')"),
        ("if inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True",
         "if inst.vpc_id is not None and inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True"),
        ("restart_retries = dict(type='int', default=0)", "restart_retries = dict(type='int', default=None)"),
        ("management.call_command('build_static')", "management.call_command('build_static', interactive=False)"),
    ]

    # -------- Randomize A/B assignment per item --------
    prompts = []
    gold_choices = []   # "A" or "B"
    ab_assignments = [] # store (A_text, B_text) to map back later

    for (before, after) in pairs:
        if random.random() < 0.5:
            A_text, B_text = before, after
            gold = "B"  # fixed is B
        else:
            A_text, B_text = after, before
            gold = "A"  # fixed is A
        ab_assignments.append((A_text, B_text))
        gold_choices.append(gold)
        prompts.append(build_inst_prompt(A_text, B_text))

    # -------- Run steered model (batch sequential) --------
    raw_outputs = malleable_model.respond_batch_sequential(prompts=prompts)

    # -------- Post-process choices & compute accuracy --------
    choices = [extract_choice_ab(r) for r in raw_outputs]
    correct = sum(1 for c, g in zip(choices, gold_choices) if c == g)
    invalid = sum(1 for c in choices if c not in ("A", "B"))
    total = len(choices)

    # Map choice back to code for inspection
    mapped_code = []
    for (choice, (A_text, B_text)) in zip(choices, ab_assignments):
        if choice == "A":
            mapped_code.append(A_text)
        elif choice == "B":
            mapped_code.append(B_text)
        else:
            mapped_code.append("")

    # -------- Pretty print --------
    for i, (pr, raw, ch, gold, (A_text, B_text)) in enumerate(zip(prompts, raw_outputs, choices, gold_choices, ab_assignments)):
        print(f"\n==== Item {i} ====")
        print("PROMPT:\n", pr.split(FEW_SHOT_AB)[-1])  # show only the final [INST] block for brevity
        print("RAW OUTPUT:", repr(raw))
        print("CHOICE:", ch, " | GOLD:", gold)
        print("A:", A_text)
        print("B:", B_text)
        print("PREDICTED CODE:", mapped_code[i])

    print("\n================ Summary ================")
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Invalid : {invalid}/{total} = {invalid/total:.2%}")

if __name__ == "__main__":
    main()

