# utils.py
import sys
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Path to the repo that contains the `activation_steering` package directory
# Adjust this if your path is different.
STEERING_ROOT = "/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"
if STEERING_ROOT not in sys.path:
    sys.path.append(STEERING_ROOT)

try:
    from activation_steering import MalleableModel, SteeringVector
except Exception:
    MalleableModel = None
    SteeringVector = None


def sanitize_completion(raw: str) -> str:
    """Strip markdown fences, tests, and explanations."""
    if "```" in raw:
        raw = raw.split("```", 1)[0]

    lines = raw.splitlines()
    kept = []

    for line in lines:
        if line.strip() == "":
            kept.append(line)
            continue
        if line.startswith("    ") or line.startswith("\t"):
            kept.append(line)
        else:
            break

    return "\n".join(kept).rstrip() if kept else raw.strip()


def generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """Run generation and post-process into a completion body."""
    ipt = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **ipt,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=tok.eos_token_id,
        )
    text = tok.decode(out[0], skip_special_tokens=True)
    body = text[len(prompt):] if text.startswith(prompt) else text
    return sanitize_completion(body)


def load_model_and_tokenizer(
    model_name: str,
    torch_dtype=torch.float16,
    device_map: str = "auto",
):
    """Load HF model + tokenizer with sane defaults for code generation."""
    tok = AutoModelForCausalLM.from_pretrained.__self__ if False else AutoTokenizer.from_pretrained(model_name)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device_map,
        torch_dtype=torch_dtype,
    )
    model.eval()
    return model, tok


def _fix_wrapped_layers_for_qwen2(mal_obj):
    """
    Make LeashLayer-wrapped decoder blocks expose attrs Qwen2 expects.
    Safe no-op on non-Qwen2 models.
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

        # Ensure attention_type is visible on the wrapper
        if not hasattr(blk, "attention_type"):
            att = getattr(
                src,
                "attention_type",
                getattr(getattr(src, "self_attn", None), "attention_type", None),
            )
            if att is not None:
                setattr(blk, "attention_type", att)

        # Mirror a few commonly-read attrs
        for attr in ("config", "hidden_size", "layer_idx"):
            if not hasattr(blk, attr) and hasattr(src, attr):
                setattr(blk, attr, getattr(src, attr))


def _parse_layer_ids(raw_layers: str, mal_model) -> list:
    """Parse --layers string into valid layer IDs for this model."""
    try:
        layers_mod = (
            mal_model.model.model.layers
            if hasattr(mal_model.model, "model")
            else mal_model.model.layers
        )
    except Exception:
        return []

    depth = len(layers_mod)
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

    # Comma-separated list of ints
    try:
        requested = sorted({int(x) for x in raw.split(",") if x.strip()})
    except Exception:
        requested = []
    layer_ids = [i for i in requested if 0 <= i < depth]
    dropped = [i for i in requested if i not in layer_ids]
    if dropped:
        print(f"[warn] Dropped out-of-range layer ids {dropped} for model with {depth} layers.")
    return layer_ids


def maybe_apply_steering(
    base_model,
    tokenizer,
    steer: bool,
    vector_path: str,
    strength: float,
    layers: str,
):
    """
    Optionally wrap the model with MalleableModel and apply a SteeringVector.
    Returns (model_for_gen, layer_ids_used or None).
    """
    if not steer:
        return base_model, None

    if MalleableModel is None or SteeringVector is None:
        raise RuntimeError("activation_steering not installed or import failed.")

    print("[INFO] Enabling steering...")
    vec = SteeringVector.load(vector_path)
    mal = MalleableModel(model=base_model, tokenizer=tokenizer)
    _fix_wrapped_layers_for_qwen2(mal)
    layer_ids = _parse_layer_ids(layers, mal)

    # Optional hidden_size sanity check
    hvec = getattr(vec, "hidden_size", None)
    hmdl = getattr(mal.model.config, "hidden_size", None)
    if hvec and hmdl and hvec != hmdl:
        print(f"[warn] Steering vector hidden_size {hvec} != model hidden_size {hmdl}.")

    mal.steer(
        behavior_vector=vec,
        behavior_layer_ids=layer_ids,
        behavior_vector_strength=strength,
    )
    print(f"[INFO] Steering applied on layers {layer_ids} with strength {strength}.")
    return mal.model, layer_ids

