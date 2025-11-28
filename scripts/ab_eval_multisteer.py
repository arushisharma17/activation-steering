#!/usr/bin/env python3
"""
A/B APR evaluation with Baseline vs Single-vector Steering vs Multi-conditioning Steering (CAST).

Usage examples
--------------
# Baseline only
python ab_eval_multisteer.py --pairs_path data/pairs.jsonl --mode baseline

# Single-vector steering
python ab_eval_multisteer.py --pairs_path data/pairs.jsonl --mode steered \
  --vector_path apr_fix_behavior --layers 20,21,22,23,24 --strength 1.2

# Multi-conditioning steering (CAST)
python ab_eval_multisteer.py --pairs_path data/pairs.jsonl --mode multisteer \
  --behavior_stem apr_fix_behavior --cond_dir condition_vectors \
  --behavior_layers 15,16,17,18,19,20,21,22,23 --strength 1.0 --rule_mode any

# Baseline vs Multi-conditioning (apples-to-apples)
python ab_eval_multisteer.py --pairs_path data/pairs.jsonl --compare \
  --mode multisteer --behavior_stem apr_fix_behavior --cond_dir condition_vectors

Notes
-----
- JSONL input requires at least keys: {"before": "...", "after": "..."} per line.
- For multi-conditioning, expect files in --cond_dir:
    <NAME>_condition_vector.svec
    optimal_condition_point_<NAME>.json  (with best_layer, best_threshold, best_direction)
- Steering uses the IBM activation-steering API:
    pip install activation-steering
"""

import os, json, random, argparse, re
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional imports (only needed for steering modes)
try:
    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.config import GlobalConfig
    from rich.console import Console
    # Avoid Rich parsing [INST] as markup
    GlobalConfig.console = Console(markup=False)
except Exception:
    MalleableModel = None
    SteeringVector = None

HF_CACHE = ""  # set to a path if you want to persist HF cache locally

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="A/B APR eval: Baseline vs Single-vector vs Multi-conditioning (CAST).")
    ap.add_argument("--pairs_path", required=True, help="Input JSONL with keys: before, after (metadata allowed).")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N usable pairs (default 0).")
    ap.add_argument("--limit", type=int, default=0, help="Use first N pairs after --start (0=all).")
    ap.add_argument("--fewshot_k", type=int, default=3, help="Few-shot examples taken from head (0=none).")

    # Model / decoding
    ap.add_argument("--model_id", default="meta-llama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--deterministic", action="store_true", help="Greedy decoding (temperature=0).")
    ap.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature when not deterministic.")
    ap.add_argument("--top_p", type=float, default=0.9, help="Top-p when not deterministic.")

    # Modes
    ap.add_argument("--mode", choices=["baseline","steered","multisteer"], default="baseline")
    ap.add_argument("--compare", action="store_true",
                    help="Run baseline and chosen steered mode back-to-back on the same prompts.")

    # Single-vector steering params
    ap.add_argument("--vector_path", default="apr_fix_behavior",
                    help="Path stem to SteeringVector for single-vector steering (loads *.svec).")
    ap.add_argument("--layers", default="27,28,29,30,31", help="Comma-separated layer ids for single-vector steering.")

    # Multi-conditioning (CAST) params
    ap.add_argument("--behavior_stem", default="apr_fix_behavior",
                    help="Stem for behavior (repair) vector (loads <stem>.svec).")
    ap.add_argument("--cond_dir", default="condition_vectors",
                    help="Dir with <name>_condition_vector.svec and optimal_condition_point_<name>.json.")
    ap.add_argument("--behavior_layers", default="15,16,17,18,19,20,21,22,23",
                    help="Comma-separated layer ids for behavior vector in multisteer.")
    ap.add_argument("--rule_mode", choices=["any","all"], default="any",
                    help="'any' to OR conditions, 'all' to AND them in the multisteer rule.")

    # Shared steering param
    ap.add_argument("--strength", type=float, default=1.2, help="Behavior vector strength.")

    # Repro/IO
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_json", default="", help="Save randomized items (few-shot header + eval prompts).")
    ap.add_argument("--load_json", default="", help="Load items for exact reproducibility.")
    ap.add_argument("--show_n", type=int, default=6, help="How many examples to print.")
    return ap.parse_args()

# ---------------- normalization helpers ----------------
_PUNCT = r"()\[\]{},.:;=+\-*/<>%&|^!~"
PUNCT_RE = re.compile(rf"\s*([{re.escape(_PUNCT)}])\s*")

def canon_one_line(s: str) -> str:
    s = (s or "").strip()
    s = " ".join(s.split())            # collapse whitespace
    s = PUNCT_RE.sub(r"\1", s)         # tighten spaces around punctuation
    return s

def iter_pairs_from_jsonl(path):
    """Yield (before, after, meta) from dataset lines that contain both."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            ex = json.loads(ln)
            before_raw = ex.get("before", "")
            after_raw  = ex.get("after", "")
            if not before_raw or not after_raw:
                continue
            before = canon_one_line(before_raw)
            after  = canon_one_line(after_raw)
            meta = {
                "project": ex.get("project",""),
                "project_url": ex.get("project_url",""),
                "commit_sha": ex.get("commit_sha",""),
                "parent_sha": ex.get("parent_sha",""),
                "file_path": ex.get("file_path",""),
                "sstub_pattern": ex.get("sstub_pattern",""),
                "likely_bug": ex.get("likely_bug", False),
                "in_function": ex.get("in_function", False),
                "diff": ex.get("diff","")
            }
            yield before, after, meta

# ---------------- prompt building ----------------
def fmt_fewshot(a, b, gold):
    return (f"[INST] You are an automated program repair system. "
            f"One of the following lines is buggy and the other is the correct fixed version.\n"
            f"Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
            f"A) {a}\nB) {b}\nAnswer: {gold} [/INST]\n\n")

def fewshot_header(fewshot_items):
    if not fewshot_items:
        return ""
    return "<s>" + "".join(fmt_fewshot(d["A"], d["B"], d["gold"]) for d in fewshot_items)

def build_eval_inst(header, A, B):
    return (header +
            "[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.\n"
            "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
            f"A) {A}\nB) {B}\nAnswer: [/INST]")

def extract_choice_ab(text):
    s = (text or "").strip().upper()
    if s.startswith("A"): return "A"
    if s.startswith("B"): return "B"
    for key in ("ANSWER:", "ANSWER IS", "CORRECT:", "CHOICE:", "ANS:"):
        i = s.find(key)
        if i != -1:
            tail = s[i+len(key):].strip()
            if tail.startswith("A"): return "A"
            if tail.startswith("B"): return "B"
    for tok in s.replace(")", " ").replace(".", " ").split():
        if tok == "A": return "A"
        if tok == "B": return "B"
    return ""

def build_items_from_file(path, seed, start, limit, k):
    rng = random.Random(seed)
    # 1) load all usable pairs
    all_pairs = list(iter_pairs_from_jsonl(path))
    if not all_pairs:
        raise ValueError("No usable (before, after) pairs found in file.")

    # 2) apply start/limit window
    if start > 0:
        all_pairs = all_pairs[start:]
    if limit and limit > 0:
        all_pairs = all_pairs[:limit]
    if not all_pairs:
        raise ValueError("Empty selection after --start/--limit filtering.")

    # 3) few-shot split
    k = max(0, min(k, len(all_pairs)))
    fewshot_raw = all_pairs[:k]
    eval_raw    = all_pairs[k:]

    # 4) randomize A/B (no overlap between few-shot and eval)
    fewshot_items = []
    for before, after, meta in fewshot_raw:
        if rng.random() < 0.5:
            A, B, gold = before, after, "B"
        else:
            A, B, gold = after, before, "A"
        fewshot_items.append({"A": A, "B": B, "gold": gold, "meta": meta})

    header = fewshot_header(fewshot_items)

    eval_items = []
    for before, after, meta in eval_raw:
        if rng.random() < 0.5:
            A, B, gold = before, after, "B"
        else:
            A, B, gold = after, before, "A"
        prompt = build_eval_inst(header, A, B)
        eval_items.append({"A": A, "B": B, "gold": gold, "prompt": prompt, "meta": meta})

    return {"fewshot": fewshot_items, "eval": eval_items}

# ---------------- runners ----------------
def _encode_for_chat(tok, model, prompt: str):
    """Handle chat template if present; else raw text encoding."""
    if hasattr(tok, "apply_chat_template"):
        messages = [{"role":"user","content":prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tok(text, return_tensors="pt").to(model.device)
    else:
        enc = tok(prompt, return_tensors="pt").to(model.device)
    return enc

def run_once(model, tok, prompts, deterministic: bool, temperature: float, top_p: float):
    outs = []
    gen_kwargs = dict(max_new_tokens=4, pad_token_id=tok.eos_token_id)
    if deterministic:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0, top_p=1.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    for p in prompts:
        enc = _encode_for_chat(tok, model, p)
        with torch.no_grad():
            gen = model.generate(**enc, **gen_kwargs)
        gen_ids = gen[0, enc["input_ids"].shape[1]:]
        outs.append(tok.decode(gen_ids, skip_special_tokens=True))
    return outs

def score_outputs(raw_outputs, items):
    preds = [extract_choice_ab(r) for r in raw_outputs]
    golds = [it["gold"] for it in items]
    acc = sum(p==g for p,g in zip(preds,golds))
    invalid = sum(p not in ("A","B") for p in preds)
    return preds, golds, acc, invalid

def pretty_print_samples(raw_outputs, items, preds, show_n):
    show = min(show_n, len(items))
    for i in range(show):
        it = items[i]
        meta = it.get("meta", {})
        print(f"\n==== Item {i} ====")
        print("PROMPT (last block):\n", it["prompt"].split("<s>")[-1])
        print("RAW OUTPUT:", repr(raw_outputs[i]))
        print("CHOICE:", preds[i], "| GOLD:", it["gold"])
        print("A:", it["A"])
        print("B:", it["B"])
        print("PREDICTED CODE:", it["A"] if preds[i]=="A" else it["B"] if preds[i]=="B" else "")
        print("META:", {k: meta.get(k) for k in ("project","file_path","sstub_pattern")})

# -------- Single-vector steering --------
def single_vector_generate(model, tok, prompts, vector_path: str, layer_ids: List[int],
                           strength: float, deterministic: bool, temperature: float, top_p: float) -> List[str]:
    if MalleableModel is None or SteeringVector is None:
        raise RuntimeError("activation_steering not available; install it or skip steering modes.")
    vec = SteeringVector.load(vector_path)
    mal = MalleableModel(model=model, tokenizer=tok)
    mal.steer(behavior_vector=vec, behavior_layer_ids=layer_ids, behavior_vector_strength=strength)

    outs = []
    gen_kwargs = dict(max_new_tokens=4, pad_token_id=tok.eos_token_id)
    if deterministic:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0, top_p=1.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    for p in prompts:
        enc = _encode_for_chat(tok, model, p)
        with torch.no_grad():
            gen = mal.model.generate(**enc, **gen_kwargs)  # mal wraps model/tokenizer
        gen_ids = gen[0, enc["input_ids"].shape[1]:]
        outs.append(tok.decode(gen_ids, skip_special_tokens=True))
    return outs

# -------- Multi-conditioning (CAST) --------
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

def _load_condition(cond_dir: str, name: str):
    """
    Load one condition's vector + gating config.
    Returns: (vec, layer_ids: List[int], threshold: float, comparator: 'larger'|'smaller')
    Accepts best_layers (list or comma-string) or best_layer (single).
    """
    vec_path  = os.path.join(cond_dir, f"{name}_condition_vector.svec")
    gate_path = os.path.join(cond_dir, f"optimal_condition_point_{name}.json")
    if not (os.path.isfile(vec_path) and os.path.isfile(gate_path)):
        raise FileNotFoundError(f"Missing files for '{name}': {vec_path} or {gate_path}")

    vec  = SteeringVector.load(vec_path)
    gate = json.load(open(gate_path, "r"))

    def pick(keys, default=None):
        for k in keys:
            if k in gate and gate[k] is not None:
                return gate[k]
        return default

    def as_int_list(val):
        # Normalize any representation into List[int]
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            out = []
            for x in val:
                try: out.append(int(x))
                except Exception: pass
            return out
        if isinstance(val, str):
            # allow "21,22,23" or "31"
            parts = [p.strip() for p in val.split(",")]
            out = []
            for p in parts:
                if not p: continue
                try: out.append(int(p))
                except Exception: pass
            return out
        # number-like
        try:
            return [int(val)]
        except Exception:
            return []

    # ---- Layers ----
    layers_val = pick(["best_layers", "best_layer_ids", "layer_ids", "best_layer", "layer", "condition_layer"])
    layer_ids = as_int_list(layers_val)
    if not layer_ids:
        # sensible fallback (common final layer index for 32-layer models)
        layer_ids = [31]

    # ---- Threshold ----
    thr_val = pick(["best_threshold", "threshold", "cond_threshold"], 0.0)
    try:
        best_threshold = float(thr_val)
    except Exception:
        best_threshold = 0.0

    # ---- Direction -> comparator ----
    dir_val = str(pick(["best_direction", "direction", "sign"], "positive")).lower()
    comparator = "larger" if dir_val.startswith(("pos", ">", "larger")) else "smaller"

    # Optional debug:
    # print(f"[cond:{name}] gate keys: {list(gate.keys())}")
    # print(f"[cond:{name}] using layers={layer_ids}, thr={best_threshold:.6f}, comp={comparator}")

    return vec, layer_ids, best_threshold, comparator



def _load_condition_orig(cond_dir: str, name: str) -> Tuple['SteeringVector', List[int], float, str]:
    vec_path  = os.path.join(cond_dir, f"{name}_condition_vector.svec")
    gate_path = os.path.join(cond_dir, f"optimal_condition_point_{name}.json")
    if not (os.path.isfile(vec_path) and os.path.isfile(gate_path)):
        raise FileNotFoundError(f"Missing files for '{name}': {vec_path} or {gate_path}")
    vec  = SteeringVector.load(vec_path)
    gate = json.load(open(gate_path, "r"))
    best_layer     = int(gate.get("best_layers"))
    best_threshold = float(gate.get("best_threshold", 0.0))
    direction      = gate.get("best_direction", "positive")
    comparator     = "larger" if direction == "positive" else "smaller"
    return vec, [best_layer], best_threshold, comparator

def multisteer_generate(model, tok, prompts, behavior_stem: str, behavior_layers: List[int],
                        strength: float, cond_dir: str, rule_mode: str,
                        deterministic: bool, temperature: float, top_p: float,
                        bugtype_keys: List[str] = BUGTYPE_KEYS) -> List[str]:
    if MalleableModel is None or SteeringVector is None:
        raise RuntimeError("activation_steering not available; install it or skip steering modes.")

    # Behavior vector
    behavior_path = f"{behavior_stem}.svec"
    if not os.path.isfile(behavior_path):
        raise FileNotFoundError(f"Behavior vector not found: {behavior_path}")
    behavior_vec = SteeringVector.load(behavior_stem)

    # Conditions
    condition_vectors, condition_layer_ids = [], []
    condition_thresholds, condition_comparators = [], []
    for name in bugtype_keys:
        try:
            vec, layers, thr, comp = _load_condition(cond_dir, name)
        except FileNotFoundError:
            print(f"  - [skip] condition {name} (files missing)")
            continue
        condition_vectors.append(vec)
        condition_layer_ids.append(layers)
        condition_thresholds.append(thr)
        condition_comparators.append(comp)

    if not condition_vectors:
        raise RuntimeError("No condition vectors found under --cond_dir.")

    # Rule string: "if C1 or C2 ... then B1" (or "and")
    all_C = [f"C{i}" for i in range(1, len(condition_vectors) + 1)]
    joiner = " or " if rule_mode == "any" else " and "
    rule = f"if {joiner.join(all_C)} then B1"

    mal = MalleableModel(model=model, tokenizer=tok)
    mal.multisteer(
        behavior_vectors=[behavior_vec],                  # B1
        behavior_layer_ids=[behavior_layers],
        behavior_vector_strengths=[strength],
        condition_vectors=condition_vectors,              # C1..Ck
        condition_layer_ids=condition_layer_ids,
        condition_vector_thresholds=condition_thresholds,
        condition_comparator_threshold_is=condition_comparators,
        rules=[rule],
    )

    outs = []
    gen_kwargs = dict(max_new_tokens=4, pad_token_id=tok.eos_token_id)
    if deterministic:
        gen_kwargs.update(dict(do_sample=False, temperature=0.0, top_p=1.0))
    else:
        gen_kwargs.update(dict(do_sample=True, temperature=temperature, top_p=top_p))

    for p in prompts:
        enc = _encode_for_chat(tok, model, p)
        with torch.no_grad():
            gen = mal.model.generate(**enc, **gen_kwargs)
        gen_ids = gen[0, enc["input_ids"].shape[1]:]
        outs.append(tok.decode(gen_ids, skip_special_tokens=True))
    return outs

# ---------------- main ----------------
def main():
    args = parse_args()

    # Build or load randomized items (ensures apples-to-apples across runs)
    if args.load_json:
        with open(args.load_json, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        items = build_items_from_file(
            path=args.pairs_path,
            seed=args.seed,
            start=args.start,
            limit=args.limit,
            k=args.fewshot_k
        )
        if args.save_json:
            with open(args.save_json, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

    eval_items = items["eval"]
    if not eval_items:
        raise ValueError("No eval items (maybe all used for few-shot?). Reduce --fewshot_k or widen selection.")

    # Load model & tokenizer once
    print(f"Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, cache_dir=HF_CACHE
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=HF_CACHE)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    prompts = [it["prompt"] for it in eval_items]

    # Baseline runner
    def run_baseline_flow():
        print("\n>>> Running BASELINE (no steering)")
        raw_b = run_once(model, tok, prompts, args.deterministic, args.temperature, args.top_p)
        preds_b, golds, acc_b, inv_b = score_outputs(raw_b, eval_items)
        pretty_print_samples(raw_b, eval_items, preds_b, args.show_n)
        print("\n[Baseline] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
            acc_b, len(eval_items), acc_b/len(eval_items), inv_b, len(eval_items), inv_b/len(eval_items)
        ))
        return acc_b, inv_b

    # Single-vector steering runner
    def run_steered_flow():
        if MalleableModel is None:
            raise RuntimeError("activation_steering not available; install it or skip steering.")
        print("\n>>> Running STEERED (single behavior vector)")
        layer_ids = [int(x) for x in args.layers.split(",") if x.strip()]
        raw_s = single_vector_generate(
            model, tok, prompts,
            vector_path=args.vector_path,
            layer_ids=layer_ids,
            strength=args.strength,
            deterministic=args.deterministic,
            temperature=args.temperature,
            top_p=args.top_p
        )
        preds_s, golds, acc_s, inv_s = score_outputs(raw_s, eval_items)
        pretty_print_samples(raw_s, eval_items, preds_s, args.show_n)
        print("\n[Steered ] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
            acc_s, len(eval_items), acc_s/len(eval_items), inv_s, len(eval_items), inv_s/len(eval_items)
        ))
        return acc_s, inv_s

    # Multi-conditioning steering runner
    def run_multisteer_flow():
        if MalleableModel is None:
            raise RuntimeError("activation_steering not available; install it or skip multisteer.")
        print("\n>>> Running MULTISTEER (multi-conditioning CAST)")
        behavior_layers = [int(x) for x in args.behavior_layers.split(",") if x.strip()]
        raw_m = multisteer_generate(
            model, tok, prompts,
            behavior_stem=args.behavior_stem,
            behavior_layers=behavior_layers,
            strength=args.strength,
            cond_dir=args.cond_dir,
            rule_mode=args.rule_mode,
            deterministic=args.deterministic,
            temperature=args.temperature,
            top_p=args.top_p
        )
        preds_m, golds, acc_m, inv_m = score_outputs(raw_m, eval_items)
        pretty_print_samples(raw_m, eval_items, preds_m, args.show_n)
        print("\n[Multistr] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
            acc_m, len(eval_items), acc_m/len(eval_items), inv_m, len(eval_items), inv_m/len(eval_items)
        ))
        return acc_m, inv_m

    # Orchestrate runs
    if args.compare:
        acc_b, inv_b = run_baseline_flow()
        if args.mode == "steered":
            acc_s, inv_s = run_steered_flow()
            label = "Steered"
            acc_x, inv_x = acc_s, inv_s
        elif args.mode == "multisteer":
            acc_m, inv_m = run_multisteer_flow()
            label = "Multistr"
            acc_x, inv_x = acc_m, inv_m
        else:
            raise ValueError("--compare requires --mode steered or --mode multisteer")
        n = len(eval_items)
        print("\n================ Side-by-Side Summary ================")
        print(f"Baseline: Acc {acc_b}/{n} = {acc_b/n:.2%} | Invalid {inv_b}/{n} = {inv_b/n:.2%}")
        print(f"{label:8s}: Acc {acc_x}/{n} = {acc_x/n:.2%} | Invalid {inv_x}/{n} = {inv_x/n:.2%}")
    else:
        if args.mode == "baseline":
            run_baseline_flow()
        elif args.mode == "steered":
            run_steered_flow()
        else:  # multisteer
            run_multisteer_flow()

if __name__ == "__main__":
    main()

