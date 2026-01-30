import argparse
import sys
import re
from pathlib import Path
from typing import List, Dict, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ======================
#  PATHS / CONSTANTS
# ======================

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

GENERATIONS_ROOT = PROJECT_ROOT / "hej_generations"
TASKS_CSV = PROJECT_ROOT / "meta" / "java_tasks.csv"
BUGGY_ORIGINAL_DIR = PROJECT_ROOT / "data" / "buggy_original"

# Optional path to an external steering library (anonymized).
# Provide via --steering_root or env var STEERING_ROOT if steering is enabled.
DEFAULT_STEERING_ROOT = None


# ======================
#  MODEL DETECTION
# ======================

def is_codellama_model(model_id: str) -> bool:
    """
    Heuristically detect CodeLlama-style models from the HF id or path.
    """
    mid = model_id.lower()
    return ("codellama" in mid) or ("meta-llama" in mid and "code" in mid)


# ======================
#  STEERING HELPERS
# ======================

def _fix_wrapped_layers_for_qwen2(mal_obj):
    """
    Make wrapper layers expose attrs Qwen2 expects.
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


def _get_layers_module(mal_model):
    try:
        return (
            mal_model.model.model.layers
            if hasattr(mal_model.model, "model")
            else mal_model.model.layers
        )
    except Exception:
        return None


def _parse_layer_ids(raw_layers: str, mal_model) -> List[int]:
    """
    Parse a layer spec string ("all", "last:k", "0,5,10") into valid indices.
    """
    layers_mod = _get_layers_module(mal_model)
    if layers_mod is None:
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

    try:
        requested = sorted({int(x) for x in raw.split(",") if x.strip()})
    except Exception:
        requested = []

    return [i for i in requested if 0 <= i < depth]


def steering_condition_name(layers: List[int], alpha: float, vec_id: Optional[str] = None) -> str:
    """
    Build a filesystem-friendly condition name encoding layers + strength + optional vector ID.
    Example: steer-l15-20_a2p0_vcorrect
    """
    layers = sorted(layers)
    layer_spec = "l" + "-".join(str(l) for l in layers)

    alpha_str = f"{alpha:.2f}".rstrip("0").rstrip(".")
    alpha_spec = "a" + alpha_str.replace(".", "p")

    cond = f"steer-{layer_spec}_{alpha_spec}"
    if vec_id:
        cond += f"_v{vec_id}"
    return cond


def _maybe_import_steering(steering_root: Optional[str]):
    """
    Import an external steering library in an anonymized, configurable way.
    """
    if steering_root:
        if steering_root not in sys.path:
            sys.path.append(steering_root)

    try:
        # Keep module name generic in the anonymized version.
        # Users can map this to their actual package name when reproducing.
        from steering_lib import MalleableModel, SteeringVector  # type: ignore
        return MalleableModel, SteeringVector
    except Exception:
        return None, None


# ======================
#  PROMPTS & EXTRACTION
# ======================

def build_prompt_generic(task_row: Dict[str, str], buggy_code: str) -> str:
    """
    Prompt used for Qwen and other well-behaved models.
    """
    java_name = task_row["java_name"]

    return (
        "You are an expert Java developer. You must FIX the following Java class so that it passes "
        "all of its existing JUnit tests.\n\n"
        "CRITICAL STRUCTURE RULES (do not break these):\n"
        f"- Keep the EXACT SAME class name: {java_name}\n"
        "- Keep the EXACT SAME package declaration as in the buggy code. "
        "Do NOT change the package name or add a new package.\n"
        "- Keep ALL method signatures unchanged.\n"
        "- Only change the internal logic needed so the tests pass.\n"
        "- Do NOT add a main method or any extra classes.\n"
        "- Do NOT include explanations, comments, or markdown in your answer.\n"
        "- Output ONLY the corrected Java class, as valid Java source code.\n\n"
        "Here is the current buggy implementation, between <buggy> tags:\n\n"
        "<buggy>\n"
        f"{buggy_code}\n"
        "</buggy>\n\n"
        "Now write ONLY the corrected Java class code below this line:\n\n"
    )


def build_prompt_codellama(task_row: Dict[str, str], buggy_code: str) -> str:
    """
    Extra-strict prompt for CodeLlama, which tends to rewrite packages/classes.
    """
    java_name = task_row["java_name"]

    return (
        "You are an expert Java developer.\n"
        "Your ONLY task is to FIX THE BUG in the given Java class so that it passes all tests.\n\n"
        "ABSOLUTE RULES (YOU MUST OBEY THESE EXACTLY):\n"
        "1. DO NOT change the package name. Use the SAME package declaration as in the buggy code.\n"
        f"2. DO NOT change the class name. It MUST remain exactly: {java_name}\n"
        "3. DO NOT change any method signatures (name, parameters, return type, visibility).\n"
        "4. DO NOT add a main method, helper classes, or any extra top-level classes.\n"
        "5. DO NOT output any explanations, comments, or markdown. ONLY Java code.\n"
        "6. Your output must be ONE valid Java class file that can compile as-is.\n\n"
        "You may ONLY modify the internal logic of existing methods.\n\n"
        "Here is the current buggy implementation, between <buggy> tags:\n\n"
        "<buggy>\n"
        f"{buggy_code}\n"
        "</buggy>\n\n"
        "Now output ONLY the corrected Java class code, with the same package and class name:\n\n"
    )


def _pick_java_segment(text: str) -> str:
    """
    Given a string that may contain ``` fences and extra text,
    pick the best-looking Java code segment.
    """
    text = text.strip()
    fence = "`" * 3

    if fence not in text:
        return text

    parts = [p.strip() for p in text.split(fence) if p.strip()]
    if not parts:
        return ""

    for p in reversed(parts):
        if "package " in p and "class " in p:
            return p.strip()

    return max(parts, key=len).strip()


def _sanitize_java_source(body: str) -> str:
    """
    Clean up the body text into valid Java:
    - Choose best Java-looking segment if there are ``` fences.
    - Drop a leading 'java' / 'language: java' line.
    - Stop at any trailing ``` fence line.
    """
    if not body:
        return body

    body = _pick_java_segment(body)
    lines = body.splitlines()

    while lines and lines[0].strip().lower() in ("java", "language: java", "```java"):
        lines = lines[1:]

    cleaned = []
    for line in lines:
        if line.strip().startswith("```"):
            break
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def extract_answer(full_text: str, prompt: str) -> str:
    """
    - Strip the prompt prefix if present.
    - Extract and sanitize the Java class code.
    """
    if full_text.startswith(prompt):
        body = full_text[len(prompt):]
    else:
        body = full_text
    body = body.strip()
    if not body:
        return body
    return _sanitize_java_source(body)


def generate_one(
    model,
    tok,
    prompt: str,
    max_new_tokens: int = 256,
    temperature: float = 0.2,
    top_p: float = 0.95,
) -> str:
    """
    Generate a single completion and extract the Java class code.
    """
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
    full = tok.decode(out[0], skip_special_tokens=True)
    return extract_answer(full, prompt)


# ======================
#  STRUCTURE NORMALIZATION
# ======================

def extract_expected_package(buggy_code: str) -> Optional[str]:
    """
    Extract the package name from the buggy Java source, e.g.:
        package humaneval.buggy;
    Returns 'humaneval.buggy' or None.
    """
    m = re.search(r'^\s*package\s+([A-Za-z0-9_.]+)\s*;', buggy_code, flags=re.MULTILINE)
    return m.group(1) if m else None


def enforce_package_and_class(
    source: str,
    expected_package: Optional[str],
    expected_class: str,
) -> str:
    """
    Normalize the generated Java source so that:
    - package declaration matches expected_package (if provided)
    - public class name matches expected_class
    """
    code = source

    if expected_package:
        if re.search(r'^\s*package\s+[A-Za-z0-9_.]+\s*;', code, flags=re.MULTILINE):
            code = re.sub(
                r'^\s*package\s+[A-Za-z0-9_.]+\s*;',
                f'package {expected_package};',
                code,
                count=1,
                flags=re.MULTILINE,
            )
        else:
            code = f"package {expected_package};\n\n" + code.lstrip()

    code = re.sub(
        r'(public\s+class\s+)([A-Za-z_][A-Za-z0-9_]*)',
        rf'\1{expected_class}',
        code,
        count=1,
    )
    return code


# ======================
#  MAIN
# ======================

def main():
    ap = argparse.ArgumentParser(description="HumanEval-Java generation with optional activation steering (anonymized).")

    # Model & sampling
    ap.add_argument("--model", required=True, help="HF model path or identifier.")
    ap.add_argument("--n", type=int, default=1, help="Samples per task.")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top_p", type=float, default=0.95)

    # Output root (defaults to hej_generations/)
    ap.add_argument("--out_root", default=None, help="Root folder for generations (default: hej_generations).")

    # Steering options
    ap.add_argument("--steer", action="store_true", help="Enable activation steering.")
    ap.add_argument("--steering_root", default=None, help="Path to external steering library (optional).")
    ap.add_argument("--vector_path", default="behavior_vector", help="Path to SteeringVector.")
    ap.add_argument("--strength", type=float, default=2.0, help="Steering vector strength.")
    ap.add_argument("--layers", default="last:4", help="Layer spec: 'all', 'last:k', or '0,5,10,...'.")
    ap.add_argument("--vec_id", default=None, help="Optional ID for the vector in the condition dir name.")

    args = ap.parse_args()

    if not TASKS_CSV.exists():
        raise FileNotFoundError(f"Tasks CSV not found: {TASKS_CSV}")
    if not BUGGY_ORIGINAL_DIR.exists():
        raise FileNotFoundError(f"Missing buggy snapshot dir: {BUGGY_ORIGINAL_DIR}")

    out_root = Path(args.out_root) if args.out_root else GENERATIONS_ROOT
    out_root.mkdir(parents=True, exist_ok=True)

    slug = args.model.rstrip("/").split("/")[-1]
    model_root = out_root / slug

    print(f"[INFO] Project root : {PROJECT_ROOT}")
    print(f"[INFO] Tasks CSV    : {TASKS_CSV}")
    print(f"[INFO] Buggy dir    : {BUGGY_ORIGINAL_DIR}")
    print(f"[INFO] Model        : {args.model}")
    print(f"[INFO] Model slug   : {slug}")
    print(f"[INFO] Out root     : {out_root}")

    is_codellama = is_codellama_model(args.model)
    print(f"[INFO] Detected CodeLlama-like model: {is_codellama}")

    # -------- Load tokenizer + base model --------
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
    condition_dir_name = "baseline"

    # -------- Optional steering --------
    if args.steer:
        steering_root = args.steering_root or DEFAULT_STEERING_ROOT
        if steering_root is None:
            steering_root = str(Path.cwd())  # fallback: allow local editable installs

        MalleableModel, SteeringVector = _maybe_import_steering(steering_root)
        if MalleableModel is None or SteeringVector is None:
            raise RuntimeError(
                "Steering requested but steering library was not importable. "
                "Provide --steering_root or set it up as an importable package."
            )

        print("[INFO] Steering enabled.")
        vec = SteeringVector.load(args.vector_path)

        mal = MalleableModel(model=base_model, tokenizer=tok)
        _fix_wrapped_layers_for_qwen2(mal)
        layer_ids = _parse_layer_ids(args.layers, mal)

        mal.steer(
            behavior_vector=vec,
            behavior_layer_ids=layer_ids,
            behavior_vector_strength=args.strength,
        )
        model_for_gen = mal.model

        condition_dir_name = steering_condition_name(
            layers=layer_ids,
            alpha=args.strength,
            vec_id=args.vec_id,
        )
    else:
        print("[INFO] Baseline (no steering).")

    condition_root = model_root / condition_dir_name
    condition_root.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Writing generations to: {condition_root}")

    # -------- Load tasks from CSV --------
    import csv
    with open(TASKS_CSV) as f_in:
        tasks = list(csv.DictReader(f_in))

    print(f"[INFO] Found {len(tasks)} tasks.")
    print(f"[INFO] Generating {args.n} sample(s) per task...")

    total_written = 0

    for trow in tasks:
        task_id = trow["task_id"]
        java_name = trow["java_name"]

        bfile = BUGGY_ORIGINAL_DIR / f"{java_name}.java"
        if not bfile.exists():
            print(f"[WARN] Missing buggy file for {task_id}: {bfile}")
            continue

        buggy_code = bfile.read_text()
        expected_package = extract_expected_package(buggy_code)

        prompt = build_prompt_codellama(trow, buggy_code) if is_codellama else build_prompt_generic(trow, buggy_code)
        print(f"[INFO] Task {task_id} ({java_name})")

        for sample_idx in range(args.n):
            out_path = condition_root / f"{task_id}_{java_name}_s{sample_idx}.java"

            if out_path.exists():
                print(f"[SKIP] {out_path} already exists; leaving it unchanged.")
                continue

            comp = generate_one(
                model_for_gen,
                tok,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )

            fixed_code = enforce_package_and_class(
                comp.strip(),
                expected_package=expected_package,
                expected_class=java_name,
            )

            out_path.write_text(fixed_code)
            total_written += 1

    print(f"[INFO] Done. Wrote {total_written} .java files under {condition_root}")


if __name__ == "__main__":
    main()

