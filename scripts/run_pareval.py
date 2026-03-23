#!/usr/bin/env python3
"""Run the ParEval benchmark with an activation-steered model.

This script loads a HuggingFace causal LM, optionally applies activation
steering via MalleableModel + SteeringVector, and then generates code
completions for every prompt in a ParEval prompt JSON file.

The output JSON is written in exactly the format that ParEval's downstream
evaluation pipeline expects (drivers/run-all.py, analysis/metrics.py).

Typical usage
-------------
# Steered run
python scripts/run_pareval.py \
    --prompts /path/to/ParEval/prompts/generation-prompts.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --vector_path vectors/python/correctness_vector_100_qwen2-5-coder-7b-instruct_pca_center_suffix \
    --strength 1.5 \
    --layers 15,16,17,18,19,20,21,22,23 \
    --output results/steered_outputs.json \
    --do_sample --temperature 0.2 --num_samples_per_prompt 50

# Baseline run (no steering)
python scripts/run_pareval.py \
    --prompts /path/to/ParEval/prompts/generation-prompts.json \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --no_steer \
    --output results/baseline_outputs.json \
    --do_sample --temperature 0.2 --num_samples_per_prompt 50
"""

import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Activation steering imports (optional for --no_steer baseline runs)
try:
    from activation_steering import MalleableModel, SteeringVector
    from activation_steering.leash_layer import LeashLayer
except ImportError:
    MalleableModel = None
    SteeringVector = None
    LeashLayer = None


# ---------------------------------------------------------------------------
# Qwen2 compatibility fix (from ab_apr_eval.py)
# ---------------------------------------------------------------------------
def _fix_wrapped_layers_for_qwen2(mal):
    """Expose attrs that Qwen2/Qwen2.5 decoder blocks expect on LeashLayer wrappers."""
    try:
        layers = mal.model.model.layers if hasattr(mal.model, "model") else mal.model.layers
    except Exception:
        return
    for blk in layers:
        src = getattr(blk, "layer", None)
        if src is None:
            continue
        if not hasattr(blk, "attention_type"):
            att = getattr(
                src, "attention_type",
                getattr(getattr(src, "self_attn", None), "attention_type", None),
            )
            if att is not None:
                setattr(blk, "attention_type", att)
        for attr in ("config", "hidden_size", "layer_idx"):
            if not hasattr(blk, attr) and hasattr(src, attr):
                setattr(blk, attr, getattr(src, attr))


# ---------------------------------------------------------------------------
# Output cleaning utilities (ported from ParEval generate/utils.py)
# ---------------------------------------------------------------------------
def _clean_output_braces(output: str, prompt: str) -> str:
    """Remove prompt prefix and truncate at matching closing brace."""
    prompt_loc = output.find(prompt)
    if prompt_loc == -1:
        # If the exact prompt isn't found (e.g. chat template added tokens),
        # just return everything after the prompt length
        body = output[len(prompt):].strip() if len(output) > len(prompt) else output.strip()
    else:
        body = output[prompt_loc + len(prompt):].strip()

    # Temporarily prepend '{' so brace-matching starts from depth 1
    body = "{" + body
    stack = []
    index = 0
    while index < len(body):
        ch = body[index]
        if ch == "{":
            stack.append(ch)
        elif ch == "}":
            stack.pop()
            if len(stack) == 0:
                break
        index += 1
    return body[1:index + 1]


GPU_FUNCTION_NAME_RE = re.compile(r"__global__ void ([a-zA-Z0-9_]+)\(")
CPU_FUNCTION_NAME_RE = re.compile(r"\s*[a-zA-Z_]+ ([a-zA-Z0-9_]+)\(")


def _get_function_name(prompt: str) -> str:
    """Extract the function name from the last line of a prompt."""
    last_line = prompt.rstrip().splitlines()[-1]
    if "__global__" in prompt:
        m = GPU_FUNCTION_NAME_RE.match(last_line)
    else:
        m = CPU_FUNCTION_NAME_RE.match(last_line)
    return m.group(1) if m else ""


def _find_matching_brace(code: str, open_idx: int) -> int:
    count = 1
    for i in range(open_idx + 1, len(code)):
        if code[i] == "{":
            count += 1
        elif code[i] == "}":
            count -= 1
            if count == 0:
                return i
    return len(code)


def _clean_instruct_output(output: str, prompt: str, response_tag: str) -> str:
    """Clean instruct-model output: find code block, extract function body."""
    tag_loc = output.find(response_tag)
    if tag_loc == -1:
        # Fallback: try brace-matching
        return _clean_output_braces(output, prompt)
    body = output[tag_loc + len(response_tag):].strip()

    # Find code blocks
    code_blocks = re.findall(r"```\n(.*?)\n```", body, flags=re.DOTALL)
    code_blocks = [
        b.removeprefix("```").removeprefix("cpp").removeprefix("c++").removesuffix("```")
        for b in code_blocks
    ]

    func_name = _get_function_name(prompt)
    prioritized = [b for b in code_blocks if func_name and func_name in b]

    if code_blocks:
        selected = prioritized[0] if prioritized else code_blocks[0]
    elif "```" in body:
        idx = body.find("```")
        selected = body[idx:].removeprefix("```")
    else:
        selected = body

    if not func_name or func_name not in selected:
        return selected

    fn_start = selected.index(func_name)
    open_brace = selected.find("{", fn_start)
    if open_brace == -1:
        return selected
    close_brace = _find_matching_brace(selected, open_brace)
    return selected[open_brace + 1:close_brace] + "}"


# ---------------------------------------------------------------------------
# Prompt formatting helpers
# ---------------------------------------------------------------------------
def format_prompt_plain(prompt_text: str, prompted: bool) -> str:
    """Format prompt for base (non-instruct) models."""
    if prompted:
        return (
            "// filename: solutions/solution_1.cpp\n"
            "// here is the correct implementation of the coding exercise\n\n"
            + prompt_text
        )
    return prompt_text.strip()


def format_prompt_chatml(prompt_text: str) -> str:
    """Format prompt using ChatML template for instruct models."""
    func_name = _get_function_name(prompt_text)
    instruction = (
        f"Complete the following c++ function.\n"
        f"```c++{prompt_text.strip()}```\n"
        f"Write only the function {func_name} and no other code. "
        f"Enclose your solution in ```c++ and ```."
    )
    return (
        "<|im_start|>system\n"
        "You are an exceptionally intelligent coding assistant that consistently "
        "delivers accurate and reliable responses to user instructions.<|im_end|>\n"
        "<|im_start|>user\n"
        f"{instruction}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def clean_output(raw_text: str, prompt_text: str, formatted_prompt: str,
                 use_chat_template: bool) -> str:
    """Clean raw model output into just the function body."""
    if use_chat_template:
        return _clean_instruct_output(raw_text, prompt_text, "<|im_start|>assistant\n")
    return _clean_output_braces(raw_text, formatted_prompt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Run the ParEval benchmark with an activation-steered model."
    )
    # I/O
    ap.add_argument("--prompts", required=True,
                     help="Path to ParEval generation-prompts.json")
    ap.add_argument("--output", required=True,
                     help="Path to write the output JSON")
    ap.add_argument("--cache", default="",
                     help="JSONL file for intermediate result caching")
    ap.add_argument("--include_models", default="",
                     help="Comma-separated parallelism models to include (e.g. omp,serial). Empty = all.")
    ap.add_argument("--max_prompts", type=int, default=0,
                     help="Limit to first N prompts after filtering (0 = all)")
    ap.add_argument("--restart", action="store_true",
                     help="Ignore cache and regenerate everything")

    # Model
    ap.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct",
                     help="HuggingFace model ID or local path")
    ap.add_argument("--hf_token", default="",
                     help="HuggingFace API token for gated models")

    # Steering
    ap.add_argument("--vector_path", default="",
                     help="Path to .svec steering vector file")
    ap.add_argument("--strength", type=float, default=1.5,
                     help="Behavior vector strength (default: 1.5)")
    ap.add_argument("--layers", default="15,16,17,18,19,20,21,22,23",
                     help="Comma-separated layer IDs for steering")
    ap.add_argument("--no_steer", action="store_true",
                     help="Run baseline without any steering")

    # Generation
    ap.add_argument("--num_samples_per_prompt", type=int, default=50,
                     help="Number of code samples per prompt (default: 50)")
    ap.add_argument("--max_new_tokens", type=int, default=1024,
                     help="Maximum new tokens to generate (default: 1024)")
    ap.add_argument("--temperature", type=float, default=0.2,
                     help="Sampling temperature (default: 0.2)")
    ap.add_argument("--top_p", type=float, default=0.95,
                     help="Nucleus sampling top_p (default: 0.95)")
    ap.add_argument("--do_sample", action="store_true",
                     help="Enable sampling (default: greedy)")
    ap.add_argument("--batch_size", type=int, default=1,
                     help="Batch size for generation (default: 1)")

    # Prompt formatting
    ap.add_argument("--prompted", action="store_true",
                     help="Prepend solution comment prefix (StarCoder-style)")
    ap.add_argument("--use_chat_template", action="store_true",
                     help="Format prompts with ChatML template for instruct models")

    return ap.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    # ---- Validate arguments ----
    if not args.no_steer and not args.vector_path:
        print("[ERROR] --vector_path is required unless --no_steer is set.",
              file=sys.stderr)
        sys.exit(1)

    # ---- Load prompts ----
    print(f"[INFO] Loading prompts from {args.prompts}")
    with open(args.prompts, "r") as f:
        prompts = json.load(f)
    print(f"[INFO] Loaded {len(prompts)} prompts")

    if args.include_models:
        models = {m.strip() for m in args.include_models.split(",") if m.strip()}
        prompts = [p for p in prompts if p.get("parallelism_model") in models]
        print(f"[INFO] Filtered to {len(prompts)} prompts with models: {models}")

    if args.max_prompts > 0:
        prompts = prompts[:args.max_prompts]
        print(f"[INFO] Limited to first {args.max_prompts} prompts")

    # ---- Load cached responses ----
    cached_names = set()
    cached_responses = []
    if args.cache and not args.restart and os.path.exists(args.cache):
        print(f"[INFO] Restoring from cache: {args.cache}")
        with open(args.cache, "r") as f:
            for line in f:
                rec = json.loads(line)
                # Only reuse if generation settings match
                if (rec.get("temperature") == args.temperature
                        and rec.get("prompted") == args.prompted
                        and len(rec.get("outputs", [])) == args.num_samples_per_prompt):
                    key = (rec["name"], rec["parallelism_model"])
                    cached_names.add(key)
                    cached_responses.append(rec)
        print(f"[INFO] Restored {len(cached_responses)} cached responses")

    # Filter out already-cached prompts
    remaining_prompts = [
        p for p in prompts
        if (p["name"], p["parallelism_model"]) not in cached_names
    ]
    print(f"[INFO] {len(remaining_prompts)} prompts remaining after cache check")

    if not remaining_prompts:
        print("[INFO] All prompts already cached. Writing output.")
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(cached_responses, f, indent=4)
        print(f"[INFO] Output written to {args.output}")
        return

    # ---- Load model & tokenizer ----
    print(f"[INFO] Loading model: {args.model}")
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
    }
    if args.hf_token:
        load_kwargs["token"] = args.hf_token

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, token=args.hf_token or None
    )

    # Ensure pad token is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    # ---- Apply steering (unless --no_steer) ----
    if not args.no_steer:
        if MalleableModel is None or SteeringVector is None:
            print("[ERROR] activation_steering package not found. "
                  "Install it or use --no_steer.", file=sys.stderr)
            sys.exit(1)

        print(f"[INFO] Loading steering vector: {args.vector_path}")
        try:
            sv = SteeringVector.load(args.vector_path)
        except Exception:
            alt = args.vector_path + ".svec" if not args.vector_path.endswith(".svec") else args.vector_path
            print(f"[INFO] Retrying with: {alt}")
            sv = SteeringVector.load(alt)

        layer_ids = sorted({int(x) for x in args.layers.split(",") if x.strip()})
        print(f"[INFO] Steering layers: {layer_ids}, strength: {args.strength}")

        mal = MalleableModel(model=model, tokenizer=tokenizer)
        _fix_wrapped_layers_for_qwen2(mal)
        mal.steer(
            behavior_vector=sv,
            behavior_layer_ids=layer_ids,
            behavior_vector_strength=args.strength,
        )

        # Use the wrapped model for generation
        gen_model = mal.model
        device = mal.device
        print("[INFO] Steering applied successfully")
    else:
        gen_model = model
        device = next(model.parameters()).device
        print("[INFO] Running in baseline mode (no steering)")

    # ---- Generation loop ----
    responses = list(cached_responses)
    total_tokens = 0
    start_time = time.time()

    for prompt_idx, prompt_entry in enumerate(
        tqdm(remaining_prompts, desc="Generating", file=sys.stdout)
    ):
        prompt_text = prompt_entry["prompt"]

        # Format the prompt
        if args.use_chat_template:
            formatted = format_prompt_chatml(prompt_text)
        else:
            formatted = format_prompt_plain(prompt_text, args.prompted)

        outputs = []
        raw_outputs = []

        for sample_idx in range(args.num_samples_per_prompt):
            # Tokenize
            enc = tokenizer(formatted, return_tensors="pt").to(device)
            input_len = enc["input_ids"].shape[1]

            # Generate
            gen_kwargs = {
                "max_new_tokens": args.max_new_tokens,
                "do_sample": args.do_sample,
                "temperature": args.temperature if args.do_sample else None,
                "top_p": args.top_p if args.do_sample else None,
                "pad_token_id": tokenizer.eos_token_id,
            }
            # Remove None values (greedy mode doesn't use temperature/top_p)
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.no_grad():
                gen_ids = gen_model.generate(**enc, **gen_kwargs)

            # Decode
            raw_text = tokenizer.decode(gen_ids[0], skip_special_tokens=False)
            generated_text = tokenizer.decode(
                gen_ids[0][input_len:], skip_special_tokens=True
            )

            total_tokens += gen_ids.shape[1]

            raw_outputs.append(raw_text)

            # Clean output
            try:
                cleaned = clean_output(raw_text, prompt_text, formatted,
                                       args.use_chat_template)
            except Exception as e:
                # If cleaning fails, use the raw generated text as fallback
                cleaned = generated_text
                print(f"[WARN] Output cleaning failed for prompt "
                      f"'{prompt_entry['name']}' sample {sample_idx}: {e}",
                      file=sys.stderr)

            outputs.append(cleaned)

        # Reset LeashLayer state after each prompt (if steering is active)
        if not args.no_steer and LeashLayer is not None:
            LeashLayer.condition_met = defaultdict(lambda: False)
            LeashLayer.forward_calls = defaultdict(int)
            LeashLayer.condition_similarities = defaultdict(lambda: defaultdict(float))

        # Build response record (matches ParEval generate.py output format)
        response = {
            "problem_type": prompt_entry.get("problem_type", ""),
            "language": prompt_entry.get("language", ""),
            "name": prompt_entry["name"],
            "parallelism_model": prompt_entry["parallelism_model"],
            "prompt": prompt_entry["prompt"],
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": args.do_sample,
            "max_new_tokens": args.max_new_tokens,
            "prompted": args.prompted,
            "outputs": outputs,
            "raw_outputs": raw_outputs,
        }
        responses.append(response)

        # Write to cache
        if args.cache:
            os.makedirs(os.path.dirname(os.path.abspath(args.cache)), exist_ok=True)
            with open(args.cache, "a") as f:
                f.write(json.dumps(response) + "\n")

        # Periodic throughput logging
        elapsed = time.time() - start_time
        if elapsed > 0 and (prompt_idx + 1) % 10 == 0:
            tps = total_tokens / elapsed
            print(f"[INFO] Throughput: {tps:.1f} tokens/s "
                  f"({prompt_idx + 1}/{len(remaining_prompts)} prompts done)")

    # ---- Write final output ----
    elapsed = time.time() - start_time
    print(f"\n[INFO] Generated {len(responses)} responses in {elapsed:.1f}s")
    if elapsed > 0:
        print(f"[INFO] Average throughput: {total_tokens / elapsed:.1f} tokens/s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(responses, f, indent=4)
    print(f"[INFO] Output written to {args.output}")

    # ---- Summary ----
    steer_info = (f"vector={args.vector_path}, strength={args.strength}, "
                  f"layers={args.layers}") if not args.no_steer else "none"
    print(f"\n{'='*60}")
    print(f"  Model:      {args.model}")
    print(f"  Steering:   {steer_info}")
    print(f"  Prompts:    {len(prompts)}")
    print(f"  Samples:    {args.num_samples_per_prompt} per prompt")
    print(f"  Output:     {args.output}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
