#!/usr/bin/env python3
import json
import random
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Only needed for steered mode
try:
    from activation_steering import MalleableModel, SteeringVector
except Exception:
    MalleableModel = None
    SteeringVector = None

HF_CACHE = ""

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="meta-llama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--mode", type=str, choices=["baseline", "steered"], default="baseline")
    ap.add_argument("--vector_path", type=str, default="refusal_behavior_vector")
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--layers", type=str, default="27,28,29,30,31")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_json", type=str, default="", help="Path to save randomized prompts+gold")
    ap.add_argument("--load_json", type=str, default="", help="Path to load prompts+gold (overrides --seed randomization)")
    return ap.parse_args()

# ---------------- Few-shot header ----------------
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
    return (
        FEW_SHOT_AB
        + "[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.\n"
          "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
          f"A) {a_line}\n"
          f"B) {b_line}\n"
          "Answer: [/INST]"
    )

def extract_choice_ab(text: str) -> str:
    s = (text or "").strip().upper()
    if s.startswith("A"): return "A"
    if s.startswith("B"): return "B"
    for key in ("ANSWER:", "ANSWER IS", "CORRECT:", "CHOICE:"):
        idx = s.find(key)
        if idx != -1:
            tail = s[idx+len(key):].strip()
            if tail.startswith("A"): return "A"
            if tail.startswith("B"): return "B"
    for tok in s.replace(")", " ").replace(".", " ").split():
        if tok == "A": return "A"
        if tok == "B": return "B"
    return ""

def make_eval_items(seed: int):
    random.seed(seed)
    # Your evaluation pairs: (buggy_before, fixed_after)
    raw_pairs = [
        ("dns = dict()", "dns = dict(default=None, type='list')"),
        ("parts = env_var.split('=')", "parts = env_var.split('=', 1)"),
        ("security_opts = dict(type=list)", "security_opts = dict(type='list')"),
        ("if inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True",
         "if inst.vpc_id is not None and inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True"),
        ("restart_retries = dict(type='int', default=0)", "restart_retries = dict(type='int', default=None)"),
        ("management.call_command('build_static')", "management.call_command('build_static', interactive=False)"),
    ]
    items = []
    for (before, after) in raw_pairs:
        if random.random() < 0.5:
            A_text, B_text = before, after
            gold = "B"
        else:
            A_text, B_text = after, before
            gold = "A"
        prompt = build_inst_prompt(A_text, B_text)
        items.append({
            "A": A_text, "B": B_text,
            "prompt": prompt,
            "correct_choice": gold
        })
    return items

def load_or_make_items(load_path: str, seed: int):
    if load_path:
        with open(load_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return make_eval_items(seed)

def save_items(path: str, items):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

def run_baseline(model, tok, prompts):
    """Plain generation with the same prompt structure."""
    outputs = []
    for p in prompts:
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
        else:
            enc = tok(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=4,     # room for space/newline + 'A'/'B'
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = gen[0, enc["input_ids"].shape[1]:]
        outputs.append(tok.decode(gen_ids, skip_special_tokens=True))
    return outputs

def run_steered(malleable_model, prompts):
    """Use activation_steering interface (sequential batch)."""
    return malleable_model.respond_batch_sequential(prompts=prompts)

def main():
    args = parse_args()

    # ----- Build or load items (A/B randomized with gold) -----
    items = load_or_make_items(args.load_json, args.seed)
    if args.save_json:
        save_items(args.save_json, items)

    # ----- Load model -----
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=HF_CACHE,
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=HF_CACHE)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    # ----- Baseline or Steered -----
    if args.mode == "baseline":
        print(">>> Running BASELINE (no steering)")
        raw_outputs = run_baseline(model, tok, [it["prompt"] for it in items])
    else:
        if MalleableModel is None or SteeringVector is None:
            raise RuntimeError("activation_steering not available; install or switch --mode baseline")
        print(">>> Running STEERED")
        vec = SteeringVector.load(args.vector_path)
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        malleable = MalleableModel(model=model, tokenizer=tok)
        malleable.steer(
            behavior_vector=vec,
            behavior_layer_ids=layers,
            behavior_vector_strength=args.strength,
        )
        raw_outputs = run_steered(malleable, [it["prompt"] for it in items])

    # ----- Postprocess + accuracy -----
    choices = [extract_choice_ab(r) for r in raw_outputs]
    golds = [it["correct_choice"] for it in items]
    correct = sum(1 for c, g in zip(choices, golds) if c == g)
    invalid = sum(1 for c in choices if c not in ("A", "B"))
    total = len(items)

    # Map back to predicted code (handy for debugging)
    mapped_code = []
    for choice, it in zip(choices, items):
        if choice == "A": mapped_code.append(it["A"])
        elif choice == "B": mapped_code.append(it["B"])
        else: mapped_code.append("")

    # ----- Pretty print -----
    for i, (it, raw, ch) in enumerate(zip(items, raw_outputs, choices)):
        tail = it["prompt"].split(FEW_SHOT_AB)[-1]
        print(f"\n==== Item {i} ====")
        print("PROMPT:\n", tail)
        print("RAW OUTPUT:", repr(raw))
        print("CHOICE:", ch, "| GOLD:", it["correct_choice"])
        print("A:", it["A"])
        print("B:", it["B"])
        print("PREDICTED CODE:", mapped_code[i])

    print("\n================ Summary ================")
    print(f"Accuracy: {correct}/{total} = {correct/total:.2%}")
    print(f"Invalid : {invalid}/{total} = {invalid/total:.2%}")

if __name__ == "__main__":
    main()

