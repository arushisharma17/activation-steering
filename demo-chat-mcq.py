#!/usr/bin/env python3
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import MalleableModel, SteeringVector

HF_cache = ""

# ------------------ Load model ------------------
model_id = "meta-llama/CodeLlama-7b-Instruct-hf"
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float16,
    cache_dir=HF_cache,
)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=HF_cache)
# Safety: ensure pad token is set
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token = tokenizer.eos_token

# ------------------ Load steering vector ------------------
refusal_behavior_vector = SteeringVector.load("refusal_behavior_vector")

# ------------------ Wrap with MalleableModel ------------------
malleable_model = MalleableModel(model=model, tokenizer=tokenizer)

# Steer at a few late layers; adjust strength to taste
malleable_model.steer(
    behavior_vector=refusal_behavior_vector,
    behavior_layer_ids=[27, 28, 29, 30, 31],
    behavior_vector_strength=2.0,
)

# ------------------ Prompting utilities ------------------

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
    """Wrap a single A/B example as an [INST] block with an Answer: stub."""
    return (
        FEW_SHOT_AB
        + f"\n[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.\n"
          "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
          f"A) {a_line}\n"
          f"B) {b_line}\n"
          "Answer: [/INST]"
    )

def extract_choice_ab(text: str) -> str:
    """Robustly extract 'A' or 'B' from the model output."""
    s = text.strip().upper()
    # Fast-path checks
    if s.startswith("A"): return "A"
    if s.startswith("B"): return "B"
    # Look for 'Answer: A' / 'Answer is B' etc.
    for key in ["ANSWER:", "ANSWER IS", "CORRECT:", "CHOICE:"]:
        idx = s.find(key)
        if idx != -1:
            tail = s[idx+len(key):].strip()
            if tail.startswith("A"): return "A"
            if tail.startswith("B"): return "B"
    # Scan tokens/lines
    for tok in s.replace(")", " ").replace(".", " ").split():
        if tok == "A": return "A"
        if tok == "B": return "B"
    return ""  # invalid/unclear

# ------------------ Build your test prompts ------------------
pairs = [
    # (A=buggy/before, B=fixed/after) — keep A as buggy so labels remain stable
    ("dns = dict()", "dns = dict(default=None, type='list')"),
    ("parts = env_var.split('=')", "parts = env_var.split('=', 1)"),
    ("security_opts = dict(type=list)", "security_opts = dict(type='list')"),
    ("if inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True",
     "if inst.vpc_id is not None and inst.get_attribute('sourceDestCheck')['sourceDestCheck'] != source_dest_check: inst.modify_attribute('sourceDestCheck', source_dest_check); changed = True"),
    ("restart_retries = dict(type='int', default=0)", "restart_retries = dict(type='int', default=None)"),
    ("management.call_command('build_static')", "management.call_command('build_static', interactive=False)"),
]

instructions = [build_inst_prompt(a, b) for (a, b) in pairs]

# ------------------ Run steered model ------------------
# Note: activation_steering's respond_batch_sequential does not expose generation kwargs.
# We therefore rely on the prompt structure to force short answers and will post-process.
steered_raw = malleable_model.respond_batch_sequential(prompts=instructions)

# ------------------ Post-process to A/B and map to code ------------------
choices = [extract_choice_ab(r or "") for r in steered_raw]
mapped = []
for (choice, (a_line, b_line)) in zip(choices, pairs):
    if choice == "A":
        mapped.append(a_line)  # predicted buggy
    elif choice == "B":
        mapped.append(b_line)  # predicted fixed
    else:
        mapped.append("")      # invalid

print("Raw generations:\n", steered_raw)
print("Choices:", choices)
print("Mapped code:\n", mapped)

# If you also want quick accuracy against known gold (fixed=B):
correct = sum(1 for c in choices if c == "B")
total = len(choices)
invalid = sum(1 for c in choices if c not in ("A", "B"))
print(f"\nAccuracy (B is gold): {correct}/{total} = {correct/total:.2%}  |  Invalid: {invalid}")

