# gen_hefix.py
from __future__ import annotations

import argparse
import os
import time
from typing import Dict, Any, List, Tuple, Optional

import torch
from human_eval.data import write_jsonl
from humanevalfix_data import read_hefix_problems, build_prompt_generic

from utils import (
    load_model_and_tokenizer,
    maybe_apply_steering,
    sanitize_completion,
)

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _is_invalid(raw_text: str, cleaned: str, min_chars: int) -> Optional[str]:
    """
    Return a string reason if invalid, else None.
    """
    if raw_text is None:
        return "RAW_NONE"

    rt = raw_text.strip()
    if not rt:
        return "RAW_EMPTY"

    # Common failure mode you showed: "```\n```"
    # Anything that becomes empty after stripping fences is invalid.
    if rt.replace("`", "").strip() == "":
        return "RAW_ONLY_BACKTICKS"

    c = (cleaned or "").strip()
    if not c:
        return "CLEAN_EMPTY"

    if len(c) < min_chars:
        return f"CLEAN_TOO_SHORT(<{min_chars})"

    return None

def _generate_raw_and_clean(
    model,
    tok,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> Tuple[str, str]:
    """
    Generate raw decoded text (new tokens only) and cleaned completion.
    Uses the same robust slicing approach as utils.generate, but keeps raw too.
    """
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )

    gen_ids = outputs[0][input_len:]
    raw = tok.decode(gen_ids, skip_special_tokens=True)
    cleaned = sanitize_completion(raw)
    return raw, cleaned

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True, help="HF model id/path")
    ap.add_argument("--n", type=int, default=10, help="samples per task")

    ap.add_argument("--max_new_tokens", type=int, default=768)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Resampling / invalid handling
    ap.add_argument("--max_tries", type=int, default=3, help="max attempts per sample if invalid/empty")
    ap.add_argument("--min_chars", type=int, default=40, help="minimum cleaned chars to consider a completion valid")
    ap.add_argument("--retry_temp_mult", type=float, default=1.25, help="multiply temperature on each retry")

    # Steering
    ap.add_argument("--steer", action="store_true")
    ap.add_argument("--vector_path", default=None)
    ap.add_argument("--strength", type=float, default=2.0)
    ap.add_argument("--layers", default="last:4")

    # Naming/output
    ap.add_argument("--condition", default="baseline", help="directory name for this run")
    ap.add_argument("--out_dir", default="hefix_generations", help="root output dir")

    # Overwrite behavior
    ap.add_argument(
        "--no_overwrite",
        action="store_true",
        help="If set, refuse to overwrite existing samples.jsonl",
    )

    args = ap.parse_args()

    model_slug = args.model.rstrip("/").split("/")[-1]
    run_dir = os.path.join(args.out_dir, model_slug, args.condition)
    ensure_dir(run_dir)

    out_path = os.path.join(run_dir, "samples.jsonl")

    if os.path.exists(out_path) and args.no_overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Remove it or rerun without --no_overwrite."
        )

    print("[INFO] ===== GEN HEFIX =====")
    print(f"[INFO] model       = {args.model}")
    print(f"[INFO] model_slug  = {model_slug}")
    print(f"[INFO] condition   = {args.condition}")
    print(f"[INFO] steer       = {args.steer}")
    print(f"[INFO] out_path    = {out_path}")
    print(f"[INFO] max_tries   = {args.max_tries}  min_chars={args.min_chars}")

    base_model, tok = load_model_and_tokenizer(args.model)

    model_for_gen, used_layers = maybe_apply_steering(
        base_model=base_model,
        tokenizer=tok,
        steer=args.steer,
        vector_path=args.vector_path,
        strength=args.strength,
        layers=args.layers,
    )
    if used_layers is not None:
        print(f"[INFO] Steering active on layers: {used_layers}")

    problems = read_hefix_problems(split="test")
    print(f"[INFO] Loaded {len(problems)} tasks from humanevalpack/python")

    samples: List[Dict[str, Any]] = []
    t0 = time.time()

    total_invalid_final = 0
    total_retries_used = 0

    for idx, (tid, prob) in enumerate(problems.items(), start=1):
        if idx % 20 == 0:
            dt = time.time() - t0
            print(f"[INFO] {idx}/{len(problems)} tasks done ({dt:.1f}s)")

        prompt = build_prompt_generic(prob)

        for j in range(args.n):
            last_raw = ""
            last_clean = ""
            invalid_reason = None
            tries = 0

            temp = args.temperature

            for t in range(args.max_tries):
                tries = t + 1
                raw, clean = _generate_raw_and_clean(
                    model_for_gen,
                    tok,
                    prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=temp,
                    top_p=args.top_p,
                )

                reason = _is_invalid(raw, clean, min_chars=args.min_chars)
                last_raw, last_clean = raw, clean
                invalid_reason = reason

                if reason is None:
                    break  # valid

                # retry with slightly higher temperature (helps avoid immediate EOS / weird fences)
                temp = min(1.2, temp * args.retry_temp_mult)

            if invalid_reason is not None:
                total_invalid_final += 1
            total_retries_used += max(0, tries - 1)

            samples.append(
                {
                    "task_id": tid,
                    "sample_id": j,
                    "model": args.model,
                    "condition": args.condition,
                    # What you evaluate:
                    "completion": (last_clean or ""),
                    # What you debug:
                    "raw_completion": (last_raw or ""),
                    # Debug/meta:
                    "tries": tries,
                    "invalid_reason": invalid_reason,
                    "raw_char_len": len((last_raw or "").strip()),
                    "clean_char_len": len((last_clean or "").strip()),
                }
            )

    write_jsonl(out_path, samples)
    print(f"[INFO] Saved {len(samples)} samples -> {out_path}")
    print(f"[INFO] Final invalid samples (after retries): {total_invalid_final}/{len(samples)}")
    if len(samples) > 0:
        print(f"[INFO] Avg retries used: {total_retries_used/len(samples):.3f} per sample")

if __name__ == "__main__":
    main()

