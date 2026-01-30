#!/usr/bin/env python3
import json
import random
import argparse
import re
import os
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Optional: only needed for --mode steered / --compare
try:
    from activation_steering import MalleableModel, SteeringVector
except Exception:
    MalleableModel = None
    SteeringVector = None

HF_CACHE = os.environ.get("HF_CACHE", "")

#MCQ_CACHE_DIR = "mcq_cache"
#METRICS_PATH = os.path.join(MCQ_CACHE_DIR, "metrics_ab_apr.jsonl")

# Allow overriding cache root via environment (e.g., mcq_cache/steering_100/tssb)
MCQ_CACHE_DIR = os.environ.get("MCQ_CACHE_DIR", "mcq_cache")
METRICS_PATH = os.path.join(MCQ_CACHE_DIR, "metrics_ab_apr.jsonl")

# --- helper: fix LeashLayer-wrapped blocks for Qwen2/Qwen2.5 ------------------
def _fix_wrapped_layers_for_qwen2(mal):
    """
    Make LeashLayer-wrapped decoder blocks expose attrs Qwen2 expects.
    Safe no-op on non-Qwen2 models.
    """
    try:
        layers = mal.model.model.layers if hasattr(mal.model, "model") else mal.model.layers
    except Exception:
        return

    for blk in layers:
        # blk is a LeashLayer; the real block is blk.layer
        src = getattr(blk, "layer", None)
        if src is None:
            continue

        # ensure attention_type is visible on the wrapper
        if not hasattr(blk, "attention_type"):
            att = getattr(
                src,
                "attention_type",
                getattr(getattr(src, "self_attn", None), "attention_type", None),
            )
            if att is not None:
                setattr(blk, "attention_type", att)

        # also mirror a few commonly-read attrs
        for attr in ("config", "hidden_size", "layer_idx"):
            if not hasattr(blk, attr) and hasattr(src, attr):
                setattr(blk, attr, getattr(src, attr))


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="A/B APR eval using MCQ questions derived from buggy/fix pairs."
    )

    # --- MCQ construction vs reuse ---
    ap.add_argument(
        "--source_dataset",
        default="",
        help=(
            "Input JSONL with keys: before, after (metadata allowed). "
            "Used to BUILD MCQ questions when --mcq_questions is not provided."
        ),
    )
    ap.add_argument(
        "--output_mcq_questions",
        default="",
        help="If set, save constructed MCQ questions (few-shot + eval) to this JSON file.",
    )
    ap.add_argument(
        "--mcq_questions",
        default="",
        help="If set, LOAD prebuilt MCQ questions from this JSON file (skips rebuilding from source_dataset).",
    )
    ap.add_argument(
        "--build_only",
        action="store_true",
        help="Only build/save MCQ questions (no model loading, no generation).",
    )

    ap.add_argument(
        "--start",
        type=int,
        default=0,
        help="Skip the first N usable pairs from the source dataset (default 0).",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use first N pairs after --start (0=all).",
    )
    ap.add_argument(
        "--fewshot_k",
        type=int,
        default=3,
        help="Few-shot examples taken from the head of the source dataset (0=none).",
    )

    # New: subset selection over the MCQ eval set (works for built or preloaded)
    ap.add_argument(
        "--eval_start",
        type=int,
        default=0,
        help="Start index within the MCQ eval set (for quick tests; default 0).",
    )
    ap.add_argument(
        "--eval_limit",
        type=int,
        default=0,
        help="Use at most this many MCQ eval items after --eval_start (0=all).",
    )

    ap.add_argument(
        "--model_id",
        default="meta-llama/CodeLlama-7b-Instruct-hf",
        help="HF model id or local path.",
    )
    ap.add_argument(
        "--tokenizer_id",
        default="",
        help="Optional tokenizer id (defaults to --model_id if empty).",
    )

    ap.add_argument(
        "--mode",
        choices=["baseline", "steered"],
        default="baseline",
        help="Run one mode (baseline or steered). Ignored if --compare is set.",
    )
    ap.add_argument(
        "--compare",
        action="store_true",
        help="Run baseline and steered back-to-back on the exact same prompts.",
    )

    # Steering params (used when --mode steered or --compare)
    ap.add_argument(
        "--vector_path",
        default="refusal_behavior_vector",
        help="Path (or stem) to SteeringVector (e.g., 'foo' or 'foo.svec').",
    )
    ap.add_argument(
        "--strength",
        type=float,
        default=2.0,
        help="Steering strength (behavior_vector_strength).",
    )
    ap.add_argument(
        "--layers",
        default="27,28,29,30,31",
        help=(
            'Layer spec: "all", "last:k", "band:<early|mid|late|frac>:k", '
            'or comma-separated indices (e.g. "27,28,29").'
        ),
    )

    # Optional caches for reuse (avoid re-running baseline/steered)
    ap.add_argument(
        "--baseline_cache",
        default="",
        help="If set, save/load baseline raw outputs here to avoid recomputation.",
    )
    ap.add_argument(
        "--steered_cache",
        default="",
        help="If set, save/load steered raw outputs here to avoid recomputation.",
    )

    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument(
        "--show_n",
        type=int,
        default=6,
        help="How many cases to print verbosely.",
    )
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
    """
    Yield (before, after, meta) from dataset lines.

    Supports both:
      - TSSB-style:   {"before": "...", "after": "...", ...}
      - ManySStuBs:   {"buggy": "...", "fixed": "...", ...}
    """
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            ex = json.loads(ln)

            # 1) Try generic 'before' / 'after' naming (TSSB / your own preprocessed files)
            before_raw = ex.get("before", "")
            after_raw = ex.get("after", "")

            # 2) If missing, fall back to ManySStuBs 'buggy' / 'fixed'
            if (not before_raw or not after_raw):
                buggy_raw = ex.get("buggy", "")
                fixed_raw = ex.get("fixed", "")
                if buggy_raw and fixed_raw:
                    # Treat buggy as "before", fixed as "after"
                    before_raw = buggy_raw
                    after_raw = fixed_raw

            # 3) If still no usable pair, skip this example
            if not before_raw or not after_raw:
                continue

            before = canon_one_line(before_raw)
            after = canon_one_line(after_raw)

            # meta: merge TSSB-style + ManySStuBs-style fields
            meta = {
                "project": ex.get("project", ""),
                "project_url": ex.get("project_url", ""),
                "commit_sha": ex.get("commit_sha", ex.get("commit", "")),
                "parent_sha": ex.get("parent_sha", ""),
                "file_path": ex.get("file_path", ""),
                "sstub_pattern": ex.get("sstub_pattern", ""),
                "likely_bug": ex.get("likely_bug", False),
                "in_function": ex.get("in_function", False),
                "diff": ex.get("diff", ""),
                # ManySStuBs-specific
                "bug_type": ex.get("bug_type", ""),
                "source_dataset": ex.get("source_dataset", ""),
            }
            yield before, after, meta


def iter_pairs_from_jsonl_orig(path):
    """Yield (before, after, meta) from dataset lines that contain both."""
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            ex = json.loads(ln)
            before_raw = ex.get("before", "")
            after_raw = ex.get("after", "")
            if not before_raw or not after_raw:
                continue
            before = canon_one_line(before_raw)
            after = canon_one_line(after_raw)
            meta = {
                "project": ex.get("project", ""),
                "project_url": ex.get("project_url", ""),
                "commit_sha": ex.get("commit_sha", ""),
                "parent_sha": ex.get("parent_sha", ""),
                "file_path": ex.get("file_path", ""),
                "sstub_pattern": ex.get("sstub_pattern", ""),
                "likely_bug": ex.get("likely_bug", False),
                "in_function": ex.get("in_function", False),
                "diff": ex.get("diff", ""),
            }
            yield before, after, meta


# ---------------- prompt building ----------------
def fmt_fewshot(a, b, gold):
    return (
        "[INST] You are an automated program repair system. "
        "One of the following lines is buggy and the other is the correct fixed version.\n"
        "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
        f"A) {a}\nB) {b}\nAnswer: {gold} [/INST]\n\n"
    )


def fewshot_header(fewshot_items):
    if not fewshot_items:
        return ""
    return "<s>" + "".join(
        fmt_fewshot(d["A"], d["B"], d["gold"]) for d in fewshot_items
    )


def build_eval_inst(header, A, B):
    return (
        header
        + "[INST] You are an automated program repair system. One of the following lines is buggy and the other is the correct fixed version.\n"
          "Identify the CORRECT (fixed) variant. Answer ONLY with A or B. No explanation.\n\n"
          f"A) {A}\nB) {B}\nAnswer: [/INST]"
    )


def extract_choice_ab(text):
    s = (text or "").strip().upper()
    if s.startswith("A"):
        return "A"
    if s.startswith("B"):
        return "B"
    for key in ("ANSWER:", "ANSWER IS", "CORRECT:", "CHOICE:", "ANS:"):
        i = s.find(key)
        if i != -1:
            tail = s[i + len(key):].strip()
            if tail.startswith("A"):
                return "A"
            if tail.startswith("B"):
                return "B"
    for tok in s.replace(")", " ").replace(".", " ").split():
        if tok == "A":
            return "A"
        if tok == "B":
            return "B"
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
    eval_raw = all_pairs[k:]

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
        eval_items.append(
            {"A": A, "B": B, "gold": gold, "prompt": prompt, "meta": meta}
        )

    return {"fewshot": fewshot_items, "eval": eval_items}


# ---------------- cache & metrics helpers ----------------
def _sanitize_for_filename(text: str) -> str:
    return (
        text.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(",", "_")
        .replace(" ", "_")
    )


def _dataset_token_from_args(args) -> str:
    """
    Infer a short dataset token from mcq_questions or source_dataset.
    This is just for logging/summary, so it's okay if it's approximate.
    """
    if getattr(args, "mcq_questions", None):
        base = os.path.basename(args.mcq_questions)
        return os.path.splitext(base)[0]
    if getattr(args, "source_dataset", None):
        base = os.path.basename(args.source_dataset)
        return os.path.splitext(base)[0]
    return "unknown"


def _auto_cache_name(prefix: str, args) -> str:
    """
    Build an automatic cache filename that encodes dataset/model/fewshot/layers/strength/seed
    and the eval subset (eval_start/eval_limit).
    """
    ds_tok = _dataset_token_from_args(args)

    if args.model_id:
        model_tok = _sanitize_for_filename(args.model_id)
    else:
        model_tok = "unknown-model"

    layers_tok = _sanitize_for_filename(getattr(args, "layers", ""))
    strength_val = getattr(args, "strength", 0.0)
    strength_tok = str(strength_val).replace(".", "p")

    eval_start = getattr(args, "eval_start", 0) or 0
    eval_limit = getattr(args, "eval_limit", 0) or 0

    fname = (
        f"{prefix}_{ds_tok}_{model_tok}_"
        f"k{args.fewshot_k}_L{layers_tok}_a{strength_tok}_seed{args.seed}_"
        f"es{eval_start}_el{eval_limit}.json"
    )

    return os.path.join(MCQ_CACHE_DIR, fname)


def log_run_metrics(
    run_kind: str,  # "baseline" or "steered"
    args,
    accuracy: float,
    invalid_rate: float,
    correct: int,
    invalid: int,
    total: int,
    cache_path: str,
    vector_path: str = "",
):
    """
    Append a single JSON record for this run to METRICS_PATH.
    """
    rec = {
        "timestamp": time.time(),
        "run_kind": run_kind,                         # "baseline" or "steered"
        "dataset_token": _dataset_token_from_args(args),
        "source_dataset": getattr(args, "source_dataset", ""),
        "mcq_questions": getattr(args, "mcq_questions", ""),
        "model_id": getattr(args, "model_id", ""),
        "mode": "baseline" if run_kind == "baseline" else "steered",
        "fewshot_k": getattr(args, "fewshot_k", None),
        "layers": getattr(args, "layers", ""),
        "strength": getattr(args, "strength", None),
        "seed": getattr(args, "seed", None),
        "eval_start": getattr(args, "eval_start", None),
        "eval_limit": getattr(args, "eval_limit", None),
        "accuracy": accuracy,
        "invalid_rate": invalid_rate,
        "correct": correct,
        "invalid": invalid,
        "total": total,
        "cache_path": cache_path,
        "vector_path": vector_path,
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


# ---------------- runners ----------------
def run_once(model, tok, prompts):
    outs = []
    for p in prompts:
        if hasattr(tok, "apply_chat_template"):
            messages = [{"role": "user", "content": p}]
            text = tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            enc = tok(text, return_tensors="pt").to(model.device)
        else:
            enc = tok(p, return_tensors="pt").to(model.device)

        with torch.no_grad():
            gen = model.generate(
                **enc,
                max_new_tokens=4,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
            )
        gen_ids = gen[0, enc["input_ids"].shape[1]:]
        outs.append(tok.decode(gen_ids, skip_special_tokens=True))
    return outs


def score_outputs(raw_outputs, items):
    preds = [extract_choice_ab(r) for r in raw_outputs]
    golds = [it["gold"] for it in items]
    acc = sum(p == g for p, g in zip(preds, golds))
    invalid = sum(p not in ("A", "B") for p in preds)
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
        print(
            "PREDICTED CODE:",
            it["A"] if preds[i] == "A" else it["B"] if preds[i] == "B" else "",
        )
        print(
            "META:",
            {k: meta.get(k) for k in ("project", "file_path", "sstub_pattern")},
        )


# ---------------- main ----------------
def main():
    args = parse_args()

    # ------------------------------------------------------------------
    # Build or load MCQ questions (few-shot + eval) for A/B APR
    # ------------------------------------------------------------------
    if args.mcq_questions:
        # Reuse a prebuilt MCQ questions file
        with open(args.mcq_questions, "r", encoding="utf-8") as f:
            items = json.load(f)
    else:
        # Build from a source dataset of (before, after) pairs
        if not args.source_dataset:
            raise ValueError(
                "You must provide either --mcq_questions (prebuilt MCQ file) "
                "or --source_dataset (JSONL of before/after pairs)."
            )

        items = build_items_from_file(
            path=args.source_dataset,
            seed=args.seed,
            start=args.start,
            limit=args.limit,
            k=args.fewshot_k,
        )

        # Optionally save the constructed MCQ questions to a JSON file
        if args.output_mcq_questions:
            with open(args.output_mcq_questions, "w", encoding="utf-8") as f:
                json.dump(items, f, ensure_ascii=False, indent=2)

    eval_items = items["eval"]
    if not eval_items:
        raise ValueError(
            "No eval items (maybe all used for few-shot?). Reduce --fewshot_k or widen selection."
        )

    # Apply eval subset selection for quick tests
    es = max(0, args.eval_start)
    if args.eval_limit and args.eval_limit > 0:
        ee = es + args.eval_limit
    else:
        ee = len(eval_items)

    if es or args.eval_limit:
        if es >= len(eval_items):
            raise ValueError(
                f"eval_start {es} is beyond number of eval items {len(eval_items)}."
            )
        eval_items = eval_items[es:ee]
        if not eval_items:
            raise ValueError(
                "Empty eval subset after applying --eval_start/--eval_limit."
            )

    # If we only want to build/save MCQ questions, stop before loading models
    if args.build_only:
        n_fewshot = len(items.get("fewshot", []))
        n_eval = len(items.get("eval", []))
        print(
            f"[INFO] Built MCQ questions with {n_fewshot} few-shot "
            f"and {n_eval} eval items."
        )
        if not (args.output_mcq_questions or args.mcq_questions):
            print(
                "[WARN] --build_only used without --output_mcq_questions or --mcq_questions; "
                "nothing was saved for reuse."
            )
        return

    # Load model & tokenizer once
    tok_id = args.tokenizer_id or args.model_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        dtype=torch.float16,
        cache_dir=HF_CACHE or None,
    )
    tok = AutoTokenizer.from_pretrained(tok_id, cache_dir=HF_CACHE or None)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token

    prompts = [it["prompt"] for it in eval_items]

    # make sure cache dir (and metrics dir) exist
    os.makedirs(MCQ_CACHE_DIR, exist_ok=True)

    # ---------------- baseline flow (with optional cache) ----------------
    def run_baseline_flow():
        print("\n>>> Running BASELINE (no steering)")

        cache_path = args.baseline_cache or _auto_cache_name("baseline", args)
        raw_b = None

        if os.path.exists(cache_path):
            print(f"[INFO] Loading baseline outputs from cache: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            raw_b = cache.get("raw_outputs", None)

        if raw_b is None:
            raw_b = run_once(model, tok, prompts)
            print(f"[INFO] Saving baseline outputs to cache: {cache_path}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw_outputs": raw_b}, f, ensure_ascii=False, indent=2)

        preds_b, golds, acc_b, inv_b = score_outputs(raw_b, eval_items)
        pretty_print_samples(raw_b, eval_items, preds_b, args.show_n)
        accuracy = acc_b / len(eval_items)
        invalid_rate = inv_b / len(eval_items)
        print(
            "\n[Baseline] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
                acc_b,
                len(eval_items),
                accuracy,
                inv_b,
                len(eval_items),
                invalid_rate,
            )
        )

        log_run_metrics(
            run_kind="baseline",
            args=args,
            accuracy=accuracy,
            invalid_rate=invalid_rate,
            correct=acc_b,
            invalid=inv_b,
            total=len(eval_items),
            cache_path=cache_path,
            vector_path="",
        )

        return acc_b, inv_b

    def _load_steering_vector(path_str: str):
        """Load SteeringVector, allowing path with or without .svec suffix."""
        if SteeringVector is None:
            raise RuntimeError(
                "activation_steering not available; install it or skip --mode steered/--compare."
            )
        try:
            return SteeringVector.load(path_str)
        except Exception:
            if not path_str.endswith(".svec"):
                alt = path_str + ".svec"
                print(f"[info] Failed to load '{path_str}', trying '{alt}'")
                return SteeringVector.load(alt)
            raise

    # ---------------- steered flow (with cache + batching) ----------------
    def run_steered_flow():
        if MalleableModel is None or SteeringVector is None:
            raise RuntimeError(
                "activation_steering not available; install it or skip --mode steered/--compare."
            )

        print("\n>>> Running STEERED")

        cache_path = args.steered_cache or _auto_cache_name("steered", args)
        raw_s = None

        if os.path.exists(cache_path):
            print(f"[INFO] Loading steered outputs from cache: {cache_path}")
            with open(cache_path, "r", encoding="utf-8") as f:
                cache = json.load(f)
            raw_s = cache.get("raw_outputs", None)

        if raw_s is None:
            vec = _load_steering_vector(args.vector_path)

            mal = MalleableModel(model=model, tokenizer=tok)
            _fix_wrapped_layers_for_qwen2(mal)

            # --- Determine model depth ---
            layers_mod = mal.model.model.layers if hasattr(mal.model, "model") else mal.model.layers
            depth = len(layers_mod)

            # -------------------------------
            # Parse args.layers into layer_ids
            # supports:
            #   "all"
            #   "last:k"
            #   "band:<early|mid|late>:k"
            #   "band:<fraction>:k" e.g., band:0.3:6
            #   "i,j,k" explicit indices
            # -------------------------------
            raw = (args.layers or "").strip()

            if raw.lower() == "all":
                layer_ids = list(range(depth))

            elif raw.lower().startswith("last:"):
                # last:k  => last k layers
                try:
                    k = int(raw.split(":", 1)[1])
                except Exception:
                    k = 4
                k = max(1, min(k, depth))
                layer_ids = list(range(depth - k, depth))

            elif raw.lower().startswith("band:"):
                # band:<early|mid|late>:k
                # band:<fraction in [0,1]>:k  (e.g., band:0.3:6)
                parts = raw.split(":")
                if len(parts) < 2:
                    raise ValueError(
                        f"Invalid band spec '{raw}' (expected band:<early|mid|late|frac>[:k])"
                    )

                band_name = parts[1]
                if len(parts) >= 3 and parts[2].strip():
                    try:
                        width = int(parts[2])
                    except Exception:
                        width = 6
                else:
                    width = 6

                def _band_indices(band_name: str, width: int, depth: int):
                    band_l = band_name.lower()

                    # Named bands
                    if band_l == "early":
                        center_frac = 0.2
                    elif band_l == "mid":
                        center_frac = 0.5
                    elif band_l == "late":
                        center_frac = 0.8
                    else:
                        # Try to interpret as numeric fraction in [0,1]
                        try:
                            center_frac = float(band_name)
                        except Exception as e:
                            raise ValueError(
                                f"Unknown band '{band_name}' (expected early|mid|late or float in [0,1])"
                            ) from e
                        if not (0.0 <= center_frac <= 1.0):
                            raise ValueError(
                                f"Band fraction '{band_name}' out of range; expected 0.0 <= frac <= 1.0"
                            )

                    width_clamped = max(1, min(width, depth))
                    # center is in [0, depth-1]
                    center = int(round(center_frac * (depth - 1)))
                    start = max(0, center - width_clamped // 2)
                    end = min(depth, start + width_clamped)
                    # ensure exact width if possible
                    start = max(0, end - width_clamped)
                    return list(range(start, end))

                layer_ids = _band_indices(band_name, width, depth)
                print(f"[info] Using band '{band_name}' (width={width}) => layers {layer_ids}")

            else:
                # Explicit comma-separated indices
                try:
                    requested = sorted({int(x) for x in raw.split(",") if x.strip()})
                except Exception:
                    requested = []
                layer_ids = [i for i in requested if 0 <= i < depth]
                dropped = [i for i in requested if i not in layer_ids]
                if dropped:
                    print(
                        f"[warn] Dropped out-of-range layer ids {dropped} "
                        f"for model with {depth} layers."
                    )

            # --- Optional sanity: vector/model hidden size check ---
            hvec = getattr(vec, "hidden_size", None)
            hmdl = getattr(mal.model.config, "hidden_size", None)
            if hvec and hmdl and hvec != hmdl:
                print(
                    f"[warn] Steering vector hidden_size {hvec} != model hidden_size {hmdl}. "
                    f"Vector may not be compatible."
                )

            # --- Apply steering and run in batches ---
            mal.steer(
                behavior_vector=vec,
                behavior_layer_ids=layer_ids,
                behavior_vector_strength=args.strength,
            )

            BATCH_SIZE = 16
            raw_s = []
            total = len(prompts)
            for i in range(0, total, BATCH_SIZE):
                sub = prompts[i: i + BATCH_SIZE]
                out = mal.respond_batch_sequential(prompts=sub)
                raw_s.extend(out)
                print(f"[steered] finished {len(raw_s)}/{total} prompts", flush=True)

            print(f"[INFO] Saving steered outputs to cache: {cache_path}")
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({"raw_outputs": raw_s}, f, ensure_ascii=False, indent=2)

        preds_s, golds, acc_s, inv_s = score_outputs(raw_s, eval_items)
        pretty_print_samples(raw_s, eval_items, preds_s, args.show_n)
        accuracy = acc_s / len(eval_items)
        invalid_rate = inv_s / len(eval_items)
        print(
            "\n[Steered ] Accuracy: {}/{} = {:.2%} | Invalid: {}/{} = {:.2%}".format(
                acc_s,
                len(eval_items),
                accuracy,
                inv_s,
                len(eval_items),
                invalid_rate,
            )
        )

        log_run_metrics(
            run_kind="steered",
            args=args,
            accuracy=accuracy,
            invalid_rate=invalid_rate,
            correct=acc_s,
            invalid=inv_s,
            total=len(eval_items),
            cache_path=cache_path,
            vector_path=args.vector_path,
        )

        return acc_s, inv_s

    # Run baseline / steered / both
    if args.compare:
        acc_b, inv_b = run_baseline_flow()
        acc_s, inv_s = run_steered_flow()
        print("\n================ Side-by-Side Summary ================")
        n = len(eval_items)
        print(
            f"Baseline: Acc {acc_b}/{n} = {acc_b/n:.2%} | Invalid {inv_b}/{n} = {inv_b/n:.2%}"
        )
        print(
            f"Steered : Acc {acc_s}/{n} = {acc_s/n:.2%} | Invalid {inv_s}/{n} = {inv_s/n:.2%}"
        )
    else:
        if args.mode == "baseline":
            run_baseline_flow()
        else:
            run_steered_flow()


if __name__ == "__main__":
    main()

