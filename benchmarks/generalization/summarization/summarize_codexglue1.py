#!/usr/bin/env python3
import argparse
import json
import os
import sys

from datasets import load_dataset
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# -----------------------
# Optional steering setup (portable)
# -----------------------
# Option A (recommended): install your steering library so it is importable.
# Option B: set STEERING_ROOT to a directory containing the steering package.
#
#   export STEERING_ROOT=/path/to/steering/library
#
STEERING_ROOT = os.environ.get("STEERING_ROOT")
if STEERING_ROOT and STEERING_ROOT not in sys.path:
    sys.path.append(STEERING_ROOT)

try:
    # Generic import name for anonymized submission; adapt in artifact if needed.
    from steering_lib import MalleableModel, SteeringVector  # type: ignore
except Exception:
    MalleableModel = None
    SteeringVector = None


def build_prompt(code: str) -> str:
    return (
        "You are a helpful assistant that writes concise docstrings.\n\n"
        "### Code:\n"
        f"{code}\n\n"
        "### Summary:"
    )


def generate_summary(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 64,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tok.eos_token_id,
        )

    text = tok.decode(output[0], skip_special_tokens=True)
    return text[len(prompt):].strip() if text.startswith(prompt) else text.strip()


def _get_num_layers(model) -> int:
    n = getattr(model.config, "num_hidden_layers", None)
    if isinstance(n, int) and n > 0:
        return n

    for attr in ("model", "transformer"):
        sub = getattr(model, attr, None)
        if sub is not None:
            layers = getattr(sub, "layers", None) or getattr(sub, "h", None)
            if layers is not None:
                return len(layers)

    raise ValueError("Could not determine number of layers for steering.")


def _parse_layer_ids(raw_layers: str, base_model) -> list:
    depth = _get_num_layers(base_model)
    raw = (raw_layers or "").strip()

    if raw.lower() == "all":
        return list(range(depth))

    if raw.lower().startswith("last:"):
        try:
            k = int(raw.split(":", 1)[1])
        except Exception:
            k = 4
        k = max(1, min(k, depth))
        return list(range(depth - k, depth))

    try:
        requested = sorted({int(x) for x in raw.split(",") if x.strip()})
    except Exception:
        requested = []

    layer_ids = [i for i in requested if 0 <= i < depth]
    dropped = [i for i in requested if i not in layer_ids]
    if dropped:
        print(f"[WARN] Dropped out-of-range layer ids {dropped} for model with {depth} layers.")
    return layer_ids


def _fix_wrapped_layers_for_qwen2(mal_obj):
    """
    Make wrapper decoder blocks expose attrs some model implementations expect.
    Safe no-op on models that do not need it.
    """
    try:
        layers_mod = (
            mal_obj.model.model.layers
            if hasattr(mal_obj.model, "model")
            else mal_obj.model.layers
        )
    except Exception:
        return

    for blk in layers_mod:
        src = getattr(blk, "layer", None)
        if src is None:
            continue

        if not hasattr(blk, "attention_type"):
            att = getattr(
                src,
                "attention_type",
                getattr(getattr(src, "self_attn", None), "attention_type", None),
            )
            if att is not None:
                setattr(blk, "attention_type", att)

        for attr in ("config", "hidden_size", "layer_idx"):
            if not hasattr(blk, attr) and hasattr(src, attr):
                setattr(blk, attr, getattr(src, attr))


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True, help="HF model name or local path.")
    parser.add_argument(
        "--lang",
        required=True,
        choices=["python", "java", "go", "javascript", "php", "ruby"],
        help="CodeXGLUE language/config name.",
    )
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"])
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max examples.")
    parser.add_argument("--max_new_tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=0.95)

    # Steering options (optional)
    parser.add_argument("--steer", action="store_true", help="Enable activation steering.")
    parser.add_argument("--vector_path", default=None, help="Path to steering vector file.")
    parser.add_argument("--layers", default="last:4", help="Layer spec: 'all', 'last:k', or comma-separated ids.")
    parser.add_argument("--strength", type=float, default=2.0, help="Steering strength.")

    args = parser.parse_args()

    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading dataset (lang={args.lang}, split={args.split})...")
    ds = load_dataset("google/code_x_glue_ct_code_to_text", args.lang, split=args.split)
    if args.limit:
        ds = ds.select(range(args.limit))
        print(f"[INFO] Using limit={args.limit}")
    print(f"[INFO] Loaded {len(ds)} examples.")

    print(f"[INFO] Loading model: {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    base_model.eval()

    model_for_gen = base_model

    if args.steer:
        if MalleableModel is None or SteeringVector is None:
            raise RuntimeError(
                "Steering requested but steering library not available. "
                "Install it or set STEERING_ROOT so it can be imported."
            )
        if not args.vector_path:
            raise ValueError("--vector_path is required when using --steer")

        print(f"[INFO] Enabling steering with vector: {args.vector_path}")
        vec = SteeringVector.load(args.vector_path)
        mal = MalleableModel(model=base_model, tokenizer=tok)

        _fix_wrapped_layers_for_qwen2(mal)

        layer_ids = _parse_layer_ids(args.layers, base_model)
        print(f"[INFO] Applying steering on layers {layer_ids} with strength {args.strength}")

        mal.steer(
            behavior_vector=vec,
            behavior_layer_ids=layer_ids,
            behavior_vector_strength=args.strength,
        )
        model_for_gen = mal.model

    print(f"[INFO] Writing predictions → {args.out}")
    with open(args.out, "w", encoding="utf-8") as f:
        for i, row in enumerate(tqdm(ds), start=1):
            code = row["code"]
            gold = row["docstring"]
            prompt = build_prompt(code)

            pred = generate_summary(
                model_for_gen,
                tok,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            record = {
                "id": i,
                "code": code,
                "reference": gold,
                "prediction": pred,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print("[INFO] Done.")


if __name__ == "__main__":
    main()

