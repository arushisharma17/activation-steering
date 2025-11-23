#!/usr/bin/env python3
import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from activation_steering import SteeringDataset, SteeringVector


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------


def slugify(s: str) -> str:
    """
    Make a filesystem-safe slug from a string:
    - lowercased
    - / and \ -> _
    - non [a-z0-9_]+ -> -
    - collapse repeated dashes/underscores
    """
    s = s.strip().lower()
    s = s.replace("/", "_").replace("\\", "_")
    s = re.sub(r"[^a-z0-9_]+", "-", s)
    s = re.sub(r"-{2,}", "-", s)
    s = re.sub(r"_{2,}", "_", s)
    return s.strip("-_")


def parse_last_tokens(x) -> int:
    """
    Convert --last-tokens:
      - integer string -> int
      - 'suffix-only'  -> -1 (special mode: use only suffix tokens)
    """
    x = str(x).strip().lower()
    if x == "suffix-only":
        return -1
    try:
        return int(x)
    except Exception:
        raise ValueError("Invalid --last-tokens value: use int or 'suffix-only'.")


def derive_default_out_path(
    pairs_path: Path,
    model_id: str,
    method: str,
    last_tokens_val: int,
    dataset_name: str | None = None,
) -> Path:
    """
    Build a default filename like:
      vectors/<lang>/behavior_manysstubs_apr_manysstubs_pairs_3k_qwen2_5_coder_7b_pca_center_suffix.svec

    Language bucket is inferred from dataset_name:
      - 'manysstubs', 'manysstubs4j' -> java
      - 'tssb', 'tssb_data_3m'       -> python
      - otherwise                    -> python (default)
    """
    data_slug = slugify(pairs_path.stem)
    model_name = Path(model_id).name or model_id
    model_slug = slugify(model_name)
    method_slug = slugify(method)
    ds_slug = slugify(dataset_name) if dataset_name else "apr"

    lt_tag = "suffix" if last_tokens_val == -1 else f"lt{last_tokens_val}"

    fname = f"behavior_{ds_slug}_{data_slug}_{model_slug}_{method_slug}_{lt_tag}.svec"

    ds_lower = (dataset_name or "").lower()
    if ds_lower in {"manysstubs", "manysstubs4j"}:
        lang_dir = "java"
    elif ds_lower in {"tssb", "tssb_data_3m"}:
        lang_dir = "python"
    else:
        # Default bucket; adjust if you add more datasets later
        lang_dir = "python"

    out_dir = Path("vectors") / lang_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / fname


# ---------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------


def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Train a behavior (steering) vector from contrastive APR pairs "
            "stored as correct/buggy responses in a JSON file."
        )
    )

    # Data
    ap.add_argument(
        "--pairs-json",
        required=True,
        help=(
            "Path to JSON with keys 'correct_responses' and 'buggy_responses' "
            "(e.g., apr_manysstubs_pairs_3k.json or apr_tssb_pairs_3k.json)."
        ),
    )
    ap.add_argument(
        "--max-examples",
        type=int,
        default=3000,
        help="Maximum number of contrastive pairs to use (0 = all, default: 3000).",
    )
    ap.add_argument(
        "--dataset-name",
        default="",
        help="Optional dataset name for auto-naming (e.g., 'tssb', 'manysstubs').",
    )

    # Model
    ap.add_argument(
        "--model-id",
        required=True,
        help="Model identifier or local path (e.g., 'Qwen/Qwen2.5-Coder-7B-Instruct').",
    )
    ap.add_argument(
        "--tokenizer-id",
        default="",
        help="Tokenizer identifier (defaults to --model-id if empty).",
    )
    ap.add_argument(
        "--hf-cache",
        default="",
        help="Optional HuggingFace cache dir.",
    )

    # Steering
    ap.add_argument(
        "--method",
        default="pca_center",
        help="SteeringVector training method (default: 'pca_center').",
    )
    ap.add_argument(
        "--last-tokens",
        default="suffix-only",
        help=(
            "How many last tokens to accumulate. "
            "Use an integer (e.g., 1, 2) or 'suffix-only' to use only the "
            "buggy/fixed suffix tokens. Default: suffix-only."
        ),
    )

    # Output
    ap.add_argument(
        "--out-vector-path",
        default="",
        help=(
            "Where to save the .svec file. If omitted, a name is automatically "
            "derived into ./vectors/<lang>/ based on dataset, pairs file, model, "
            "method, and last-tokens."
        ),
    )

    return ap.parse_args()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------


def main():
    args = parse_args()

    pairs_path = Path(args.pairs_json)
    with pairs_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    correct_list = data.get("correct_responses", [])
    buggy_list = data.get("buggy_responses", [])

    if not correct_list or not buggy_list:
        raise ValueError(
            f"No 'correct_responses' or 'buggy_responses' found in {pairs_path}"
        )

    n = min(len(correct_list), len(buggy_list))
    if args.max_examples > 0:
        n = min(n, args.max_examples)

    correct_list = [c.strip() for c in correct_list[:n] if c and c.strip()]
    buggy_list = [b.strip() for b in buggy_list[:n] if b and b.strip()]

    n = min(len(correct_list), len(buggy_list))
    if n == 0:
        raise ValueError("No usable (correct, buggy) pairs after filtering.")

    correct_list = correct_list[:n]
    buggy_list = buggy_list[:n]

    print(f"[Data] Using {n} contrastive pairs from {pairs_path}")

    # Parse last-tokens
    last_tokens_val = parse_last_tokens(args.last_tokens)

    # Load model & tokenizer
    tok_id = args.tokenizer_id or args.model_id
    print(f"[Model] Loading model: {args.model_id}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16,
        cache_dir=args.hf_cache or None,
    )
    print(f"[Tokenizer] Loading tokenizer: {tok_id}")
    tokenizer = AutoTokenizer.from_pretrained(
        tok_id,
        cache_dir=args.hf_cache or None,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    # Build SteeringDataset
    # We use empty prefixes; with suffix-only, this focuses purely on buggy/fixed code.
    examples = [("", "") for _ in range(n)]
    suffixes = list(zip(buggy_list, correct_list))

    steering_ds = SteeringDataset(
        tokenizer=tokenizer,
        examples=examples,
        suffixes=suffixes,
    )
    print(f"[Dataset] Built SteeringDataset with {n} examples.")

    # Train behavior vector
    print(
        f"[Train] Training behavior vector with method={args.method}, "
        f"accumulate_last_x_tokens={last_tokens_val}"
    )
    behavior_vec = SteeringVector.train(
        model=model,
        tokenizer=tokenizer,
        steering_dataset=steering_ds,
        method=args.method,
        accumulate_last_x_tokens=last_tokens_val,
    )

    # Decide save path
    if args.out_vector_path:
        out_path = Path(args.out_vector_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = derive_default_out_path(
            pairs_path=pairs_path,
            model_id=args.model_id,
            method=args.method,
            last_tokens_val=last_tokens_val,
            dataset_name=args.dataset_name or None,
        )

    behavior_vec.save(str(out_path))
    print(f"[Save] Saved steering vector to {out_path}")


if __name__ == "__main__":
    main()

