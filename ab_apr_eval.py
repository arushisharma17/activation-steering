#!/usr/bin/env python3
import json, random, argparse, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: only needed for --mode steered / --compare
try:
    from activation_steering import MalleableModel, SteeringVector
except Exception:
    MalleableModel = None
    SteeringVector = None

HF_CACHE = ""

# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="A/B APR eval from JSONL (before/after + metadata).")
    ap.add_argument("--pairs_path", required=True, help="Input JSONL with keys: before, after (metadata allowed).")
    ap.add_argument("--start", type=int, default=0, help="Skip the first N usable pairs (default 0).")
    ap.add_argument("--limit", type=int, default=0, help="Use first N pairs after --start (0=all).")
    ap.add_argument("--fewshot_k", type=int, default=3, help="Few-shot examples taken from the head (0=none).")

    ap.add_argument("--model_id", default="meta-llama/CodeLlama-7b-Instruct-hf")
    ap.add_argument("--mode", choices=["baseline","steered"], default="baseline",
                    help="Run one mode (baseline or steered). Ignored if --compare is set.")
    ap.add_argument("--compare", action="store_true",
                    help="Run baseline and steered back-to-back on the exact same prompts.")

    # Steering params (used when --mode steered or --compare)
    ap.add_argument("--vector_path", default="refusal_behavior_vector",
                    help="Path stem to SteeringVector (e.g., 'refusal_behavior_vector' -> *.svec).")
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--layers", default="27,28,29,30,31")

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_json", default="", help="Save randomized items for reuse.")
    ap.add_argument("--load_json", default="", help="Load items (skips rebuilding/randomization).")

    ap.add_argument("--show_n", type=int, default=6, help="How many cases to print verbosely.")
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
def run_once(model, tok, prompts):
    outs = []
    for p in prompts:
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role":"user","content":p}]
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            enc = tok(text, return_tensors="pt").to(model.device)
        else:
            enc = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=4, do_sample=True, temperature=0.7, top_p=0.9,
                pad_token_id=tok.eos_token_id
            )
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

# ---------------- main ----------------
def main():
    args = parse_args()

    # Build or load randomized items (ensures baseline & steered compare on the same prompts)
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
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id, device_map="auto", torch_dtype=torch.float16, cache_dir=HF_CACHE
    )
    tok = AutoTokenizer.from_pretrained(args.model_id, cache_dir=HF_CACHE)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    prompts = [it["prompt"] for it in eval_items]

    def run_baseline_flow():
        print("\n>>> Running BASELINE (no steering)")
        raw_b = run_once(model, tok, prompts)
        preds_b, golds, acc_b, inv_b = score_outputs(raw_b, eval_items)
        pretty_print_samples(raw_b, eval_items, preds_b, args.show_n)
        print("\n[Baseline] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
            acc_b, len(eval_items), acc_b/len(eval_items), inv_b, len(eval_items), inv_b/len(eval_items)
        ))
        return acc_b, inv_b

    def run_steered_flow():
        if MalleableModel is None or SteeringVector is None:
            raise RuntimeError("activation_steering not available; install it or skip --mode steered/--compare.")
        print("\n>>> Running STEERED")
        vec = SteeringVector.load(args.vector_path)
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        mal = MalleableModel(model=model, tokenizer=tok)
        mal.steer(behavior_vector=vec, behavior_layer_ids=layers, behavior_vector_strength=args.strength)
        raw_s = mal.respond_batch_sequential(prompts=prompts)
        preds_s, golds, acc_s, inv_s = score_outputs(raw_s, eval_items)
        pretty_print_samples(raw_s, eval_items, preds_s, args.show_n)
        print("\n[Steered ] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
            acc_s, len(eval_items), acc_s/len(eval_items), inv_s, len(eval_items), inv_s/len(eval_items)
        ))
        return acc_s, inv_s

    # Run baseline / steered / both
    if args.compare:
        acc_b, inv_b = run_baseline_flow()
        acc_s, inv_s = run_steered_flow()
        print("\n================ Side-by-Side Summary ================")
        n = len(eval_items)
        print(f"Baseline: Acc {acc_b}/{n} = {acc_b/n:.2%} | Invalid {inv_b}/{n} = {inv_b/n:.2%}")
        print(f"Steered : Acc {acc_s}/{n} = {acc_s/n:.2%} | Invalid {inv_s}/{n} = {inv_s/n:.2%}")
    else:
        if args.mode == "baseline":
            run_baseline_flow()
        else:
            run_steered_flow()

if __name__ == "__main__":
    main()

