# utils.py (HumanevalFix-Python)
from __future__ import annotations

import sys
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

########################################
# Import activation_steering from repo root
########################################
STEERING_ROOT = "/lustre/hdd/LAS/jannesar-lab/arushi/activation-steering"
if STEERING_ROOT not in sys.path:
    sys.path.insert(0, STEERING_ROOT)

try:
    from activation_steering import MalleableModel, SteeringVector
except Exception:
    MalleableModel = None
    SteeringVector = None

########################################
# Typing import patch (fix List/Dict NameError)
########################################
_TYPING_NAMES = ["List", "Dict", "Tuple", "Set", "Optional", "Any", "Iterable", "Sequence"]

def ensure_typing_imports(code: str) -> str:
    """
    If the code uses typing names (List, Dict, ...) but doesn't import them,
    inject `from typing import ...` near the top.

    Heuristics:
      - Only inject if at least one typing name is used as a token.
      - Do not inject if there's already `from typing import ...` or `import typing`.
      - Insert after shebang/encoding/future imports and module docstring.
    """
    if not code or not code.strip():
        return code

    # Already imported?
    if re.search(r"^\s*from\s+typing\s+import\s+", code, flags=re.M):
        return code
    if re.search(r"^\s*import\s+typing\b", code, flags=re.M):
        return code

    used = []
    for name in _TYPING_NAMES:
        if re.search(rf"\b{name}\b", code):
            used.append(name)

    if not used:
        return code

    import_line = f"from typing import {', '.join(sorted(set(used)))}"

    lines = code.splitlines()
    i = 0

    # shebang
    if lines and lines[0].startswith("#!"):
        i = 1

    # encoding cookie in first two lines
    enc_re = re.compile(r"coding[:=]\s*[-\w.]+")
    for j in range(i, min(i + 2, len(lines))):
        if enc_re.search(lines[j]):
            i = j + 1

    # future imports
    while i < len(lines) and re.match(r"^\s*from\s+__future__\s+import\s+", lines[i]):
        i += 1

    # module docstring
    if i < len(lines):
        m = re.match(r'^\s*(?P<q>"""|\'\'\')', lines[i])
        if m:
            q = m.group("q")
            i += 1
            while i < len(lines) and q not in lines[i]:
                i += 1
            if i < len(lines):
                i += 1  # include closing line

    new_lines = lines[:i] + [import_line, ""] + lines[i:]
    return "\n".join(new_lines).rstrip() + "\n"

########################################
# Completion post-processing
########################################
def sanitize_completion(raw: str) -> str:
    """
    Return python code from a model completion.

    Guarantees:
      - If the model produced anything, tries hard to return non-empty.
      - If fenced code exists, extract the first fenced block's content.
      - Otherwise, return stripped text as-is.
      - Patch typing imports if needed (fix List/Dict NameError).
    """
    if raw is None:
        return ""

    text = raw.strip()
    if not text:
        return ""

    # Prefer fenced code (first fence)
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            inside = parts[1]
            lines = inside.splitlines()
            # drop language tag line
            if lines and lines[0].strip().lower() in {"python", "py"}:
                inside = "\n".join(lines[1:])
            inside = inside.strip()
            if inside:
                return ensure_typing_imports(inside)

    # Fallback: keep everything (APR evaluator can fail it if it's junk, but it won't be empty)
    return ensure_typing_imports(text)

def generate(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Robust generation:
      - decode ONLY newly generated tokens (avoid brittle prompt-subtraction)
      - never return empty if the model generated any visible text
      - patch typing imports if needed
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
    decoded = tok.decode(gen_ids, skip_special_tokens=True)

    cleaned = sanitize_completion(decoded)
    if cleaned.strip():
        return cleaned

    # last-resort fallback (should basically never trigger unless decoded is whitespace)
    decoded2 = decoded.strip()
    return decoded2 if decoded2 else ""

########################################
# HF loading
########################################
def load_model_and_tokenizer(
    model_name: str,
    torch_dtype=torch.float16,
    device_map: str = "auto",
):
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

########################################
# Steering helpers
########################################
def _get_decoder_layers(m):
    """
    Return the list of decoder blocks for common causal LMs (Qwen2, Llama, etc.).
    Handles:
      - base HF model objects
      - activation_steering MalleableModel.model wrappers
    """
    # Common: model.model.layers (Llama-ish)
    if hasattr(m, "model") and hasattr(m.model, "layers"):
        return m.model.layers

    # Some wrappers: model.model.model.layers
    if hasattr(m, "model") and hasattr(m.model, "model") and hasattr(m.model.model, "layers"):
        return m.model.model.layers

    # Direct: layers
    if hasattr(m, "layers"):
        return m.layers

    return None

def _patch_qwen2_wrapped_layers(mal: "MalleableModel"):
    """
    Qwen2 forward reads decoder_layer.attention_type.
    activation_steering may wrap layers with LeashLayer that doesn't expose it.

    Fix: mirror attention_type (and a couple other commonly accessed attrs)
    from the wrapped `blk.layer` (original) onto the wrapper `blk`.
    """
    layers = _get_decoder_layers(mal.model)
    if layers is None:
        return

    patched_att = 0
    patched_misc = 0

    for blk in layers:
        src = getattr(blk, "layer", None)
        if src is None:
            continue

        # attention_type
        if not hasattr(blk, "attention_type"):
            att = None
            if hasattr(src, "attention_type"):
                att = getattr(src, "attention_type")
            elif hasattr(src, "self_attn") and hasattr(src.self_attn, "attention_type"):
                att = getattr(src.self_attn, "attention_type")
            if att is not None:
                setattr(blk, "attention_type", att)
                patched_att += 1

        # mirror a few attrs that some model code paths expect
        for attr in ("config", "hidden_size", "layer_idx"):
            if not hasattr(blk, attr) and hasattr(src, attr):
                setattr(blk, attr, getattr(src, attr))
                patched_misc += 1

    if patched_att or patched_misc:
        print(f"[INFO] Patched Qwen2 wrapped layers: attention_type={patched_att}, misc_attrs={patched_misc}")

def _parse_layers_spec(raw_layers: str, model_depth: int) -> list[int]:
    """
    Supports:
      - all
      - last:k
      - comma-separated indices: "1,2,3"
      - band:ratio:k   e.g. band:0.15:6 => take k layers centered around ratio*(depth-1)
    """
    raw = (raw_layers or "").strip().lower()

    # default: last 4 layers (reasonable for many models)
    if not raw:
        return list(range(max(0, model_depth - 4), model_depth))

    if raw == "all":
        return list(range(model_depth))

    if raw.startswith("last:"):
        try:
            k = int(raw.split(":", 1)[1])
        except Exception:
            k = 4
        k = max(1, min(k, model_depth))
        return list(range(model_depth - k, model_depth))

    if raw.startswith("band:"):
        m = re.match(r"band:(\d*\.?\d+):(\d+)", raw)
        if not m:
            raise ValueError(f"Bad band spec: {raw_layers} (expected band:<ratio>:<k>)")
        ratio = float(m.group(1))
        k = int(m.group(2))

        k = max(1, min(k, model_depth))
        center = int(round(ratio * (model_depth - 1)))

        half = k // 2
        start = max(0, center - half)
        end = min(model_depth, start + k)
        start = max(0, end - k)  # adjust if we hit the end

        return list(range(start, end))

    # comma-separated ints
    out: list[int] = []
    for t in raw.split(","):
        t = t.strip()
        if not t:
            continue
        out.append(int(t))
    out = sorted(set(i for i in out if 0 <= i < model_depth))
    return out

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

    Returns:
      (model_for_gen, layer_ids_used or None)
    """
    if not steer:
        return base_model, None

    if MalleableModel is None or SteeringVector is None:
        raise RuntimeError("activation_steering not installed or import failed.")

    print("[INFO] Applying activation steering")
    vec = SteeringVector.load(vector_path)
    mal = MalleableModel(model=base_model, tokenizer=tokenizer)

    layers_list = _get_decoder_layers(mal.model)
    if layers_list is None:
        raise RuntimeError("Could not determine decoder layers for --layers parsing.")
    depth = len(layers_list)
    if depth <= 0:
        raise RuntimeError("Model depth is 0; cannot parse --layers.")

    layer_ids = _parse_layers_spec(layers, depth)

    # sanity check hidden size (optional)
    hvec = getattr(vec, "hidden_size", None)
    hmdl = getattr(getattr(mal.model, "config", None), "hidden_size", None)
    if hvec and hmdl and hvec != hmdl:
        print(f"[WARN] Steering vector hidden_size {hvec} != model hidden_size {hmdl}.")

    mal.steer(
        behavior_vector=vec,
        behavior_layer_ids=layer_ids,
        behavior_vector_strength=float(strength),
    )

    # Patch after steering wraps layers (important for Qwen2)
    _patch_qwen2_wrapped_layers(mal)

    print(f"[INFO] Steering applied on layers {layer_ids}, strength={strength}")
    return mal.model, layer_ids

