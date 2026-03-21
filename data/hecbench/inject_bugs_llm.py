#!/usr/bin/env python3
"""
Prompt an LLM to inject HPC-specific bugs into HeCBench source files.

Reads the source manifest produced by fetch_hecbench_sources.py, sends
each file (paired with each applicable bug category) to a local LLM,
and writes the results to a JSONL file.

Features:
  - Checkpoint / resume: skips (benchmark, file, bug_type) triples
    already present in the output JSONL.
  - Rate-limiting (configurable delay between calls).
  - --dry-run mode for testing without an LLM.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

from config import (
    BUG_CATEGORIES,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MAX_TOKENS,
    LLM_MODEL,
    LLM_TEMPERATURE,
    MANIFEST_PATH,
    RAW_BUGS_PATH,
    SOURCES_DIR,
    USE_LOCAL_VLLM,
)

if USE_LOCAL_VLLM:
    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError:
        print("[ERROR] USE_LOCAL_VLLM is True, but vllm (or torch) is not installed.")
        print("        Install with: pip install vllm torch")
        sys.exit(1)

# ──────────────────────────────────────────────────────────────────────
# Prompt template
# ──────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert HPC programmer specializing in OpenMP.
Your task is to introduce exactly ONE subtle bug into the provided source code.
The bug should be the kind of mistake a real programmer might make.

We will provide you with a list of possible bug types. You must review the source code, 
select the SINGLE most appropriate bug type for that specific code, and apply it.

Rules:
1. Choose exactly ONE bug type from the provided list that makes the most sense for this code.
2. Make exactly ONE change that introduces this bug type.
3. The modified code MUST still compile without errors.
4. Keep the change minimal — do not rewrite large sections.
5. Do NOT add comments that hint at the bug.
6. CRITICAL: You MUST return the ENTIRE source file, from the first `#include` to the final `}`. 
   Do NOT use ellipses like `// ... rest of code ...`. You must print every single line of the original file alongside your one targeted modification.
7. Return nothing else before the code. Do NOT wrap the code in markdown code fences (` ``` `).
8. After the complete modified file, you MUST append two lines on new lines:
   BUG_TYPE: <The exact ID of the bug type you chose>
   BUG_EXPLANATION: <A one-sentence explanation of what you changed>

CRITICAL EXCEPTION:
If, and ONLY if, you cannot find ANY realistic way to introduce one of the provided bug types into this specific source code (e.g. it has no OpenMP pragmas or shared variables you can manipulate), you may choose to pass.
To pass, do NOT output any source code at all. Instead, strictly output ONLY the following two lines:
BUG_TYPE: PASS
BUG_EXPLANATION: <Reason why none of the bug types were applicable>
"""

USER_PROMPT_TEMPLATE = """\
Available bug types to choose from:
{bug_types_list}

Action: Review the code below, select the most suitable bug type from the list above, and inject it. Return the FULL source file! If impossible, return BUG_TYPE: PASS.

Here is the original source file ({filename}, {line_count} lines):

{source_code}
"""


def format_bug_types(bug_types: list[str]) -> str:
    lines = []
    for bt in bug_types:
        info = BUG_CATEGORIES[bt]
        lines.append(f"- {bt} ({info['keyword']}): {info['ai_action']}")
    return "\n".join(lines)


def build_prompt_str(
    filename: str,
    source_code: str,
    bug_types_list: str,
) -> str:
    """Build a raw string prompt for vLLM without applying a chat template (assumes base or instruct model handles raw)."""
    user_p = USER_PROMPT_TEMPLATE.format(
        bug_types_list=bug_types_list,
        filename=filename,
        line_count=len(source_code.splitlines()),
        source_code=source_code,
    )
    return f"{SYSTEM_PROMPT}\n\n{user_p}"

def build_prompt_messages(
    filename: str,
    source_code: str,
    bug_types_list: str,
) -> list[dict[str, str]]:
    """Build the chat messages list for HTTP APIs."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_PROMPT_TEMPLATE.format(
                bug_types_list=bug_types_list,
                filename=filename,
                line_count=len(source_code.splitlines()),
                source_code=source_code,
            ),
        },
    ]


def call_llm(
    messages: list[dict[str, str]],
    *,
    base_url: str,
    api_key: str,
    model: str,
    temperature: float,
    max_tokens: int,
    retries: int = 3,
) -> str:
    """Call the OpenAI-compatible chat/completions endpoint."""
    url = f"{base_url}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    for attempt in range(1, retries + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=120)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as exc:
            print(f"  [WARN] LLM call attempt {attempt}/{retries} failed: {exc}")
            if attempt < retries:
                time.sleep(2 ** attempt)
    raise RuntimeError("LLM call failed after all retries")


def parse_response(raw: str) -> tuple[str, str, str]:
    """Split the LLM response into (buggy_code, bug_type, explanation)."""
    type_marker = "BUG_TYPE:"
    expl_marker = "BUG_EXPLANATION:"
    
    idx_type = raw.rfind(type_marker)
    idx_expl = raw.rfind(expl_marker)
    
    if idx_type == -1 or idx_expl == -1:
        return raw.strip(), "UNKNOWN", "(no structured explanation provided)"
    
    first_marker = min(idx_type, idx_expl)
    code = raw[:first_marker].strip()
    
    type_start = idx_type + len(type_marker)
    type_end = raw.find("\n", type_start) if raw.find("\n", type_start) != -1 else len(raw)
    bug_type = raw[type_start:type_end].strip()
    
    expl_start = idx_expl + len(expl_marker)
    if idx_expl > idx_type:
        explanation = raw[expl_start:].strip()
    else:
        explanation = raw[expl_start:idx_type].strip()
        
    return code, bug_type, explanation


def load_checkpoint(output_path: Path) -> set[tuple[str, str]]:
    """Load already-processed (benchmark, file) pairs."""
    done: set[tuple[str, str]] = set()
    if not output_path.exists():
        return done
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            done.add((row["benchmark"], row["source_file"]))
    return done


def dry_run_response(source_code: str, bug_types: list[str]) -> str:
    """Deterministic placeholder for --dry-run mode."""
    lines = source_code.splitlines()
    bt = bug_types[0] if bug_types else "UNKNOWN"
    # Just duplicate the first non-empty line as a trivial "mutation"
    for i, line in enumerate(lines):
        if line.strip():
            lines[i] = f"/* DRY_RUN_{bt} */ {line}"
            break
    return "\n".join(lines) + f"\nBUG_TYPE: {bt}\nBUG_EXPLANATION: [dry-run] Prefixed line with comment for {bt}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject HPC bugs via LLM")
    ap.add_argument("--manifest", type=Path, default=MANIFEST_PATH)
    ap.add_argument("--sources-dir", type=Path, default=SOURCES_DIR)
    ap.add_argument("--out", type=Path, default=RAW_BUGS_PATH)
    ap.add_argument(
        "--bug-types",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated bug type IDs (default: all from config)",
    )
    ap.add_argument("--max-samples", type=int, default=0,
                    help="Stop after this many samples (0 = unlimited)")
    ap.add_argument("--delay", type=float, default=0.5,
                    help="Seconds to wait between HTTP LLM calls (ignored for local vLLM)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use a placeholder instead of calling the LLM")
    args = ap.parse_args()

    # Load manifest
    if not args.manifest.exists():
        print(f"[ERROR] Manifest not found: {args.manifest}")
        print("       Run fetch_hecbench_sources.py first.")
        sys.exit(1)

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    bug_types = args.bug_types or list(BUG_CATEGORIES.keys())

    # Validate bug types
    for bt in bug_types:
        if bt not in BUG_CATEGORIES:
            print(f"[ERROR] Unknown bug type: {bt}")
            print(f"        Available: {', '.join(BUG_CATEGORIES)}")
            sys.exit(1)

    # Checkpoint
    done = load_checkpoint(args.out)
    print(f"[INFO] {len(done)} entries already done (checkpoint)")
    print(f"[INFO] {len(manifest)} source files × {len(bug_types)} bug types")

    count = 0
    args.out.parent.mkdir(parents=True, exist_ok=True)

    # Initialize vLLM if configured
    llm = None
    sampling_params = None
    if USE_LOCAL_VLLM and not args.dry_run:
        num_gpus = torch.cuda.device_count()
        print(f"[INFO] Initializing local vLLM engine with model: {LLM_MODEL} on {num_gpus} GPUs")
        llm = LLM(model=LLM_MODEL, tensor_parallel_size=num_gpus)
        sampling_params = SamplingParams(
            temperature=LLM_TEMPERATURE,
            max_tokens=LLM_MAX_TOKENS,
        )

    requests_to_process = []
    bug_types_str = format_bug_types(bug_types)

    # 1. Collect all valid requests
    for entry in manifest:
        bench = entry["benchmark"]
        fname = entry["source_file"]

        if (bench, fname) in done:
            continue

        src_path = args.sources_dir / fname
        if not src_path.exists():
            print(f"[WARN] Source file missing: {src_path}")
            continue

        source_code = src_path.read_text(encoding="utf-8", errors="replace")
        
        requests_to_process.append({
            "bench": bench,
            "fname": fname,
            "source_code": source_code,
        })

        if args.max_samples > 0 and len(requests_to_process) >= args.max_samples:
            break

    if not requests_to_process:
        print("[INFO] No new files to process.")
        return

    print(f"[INFO] Processing {len(requests_to_process)} files...")

    # 2. Generate responses
    import copy
    results = []

    if args.dry_run:
        for req in requests_to_process:
            raw = dry_run_response(req["source_code"], bug_types)
            results.append((req, raw))
    
    elif USE_LOCAL_VLLM and llm and sampling_params:
        # -> BATCHED vLLM INFERENCE <-
        prompts = []
        for req in requests_to_process:
            # Force diversity: Randomly give the LLM only 2 bug types to pick from
            # (Ensures it can't just pick DATA_RACE every single time)
            subset = random.sample(bug_types, min(2, len(bug_types)))
            req_bug_types_str = format_bug_types(subset)
            
            prompts.append(
                build_prompt_str(
                    filename=req["fname"],
                    source_code=req["source_code"],
                    bug_types_list=req_bug_types_str,
                )
            )
        
        # vLLM handles continuous batching natively!
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        
        for req, output in zip(requests_to_process, outputs):
            results.append((req, output.outputs[0].text))
            
    else:
        # -> HTTP API (Sequential) <-
        for req in tqdm(requests_to_process, desc="HTTP LLM calls"):
            subset = random.sample(bug_types, min(2, len(bug_types)))
            req_bug_types_str = format_bug_types(subset)

            messages = build_prompt_messages(
                filename=req["fname"],
                source_code=req["source_code"],
                bug_types_list=req_bug_types_str,
            )
            raw = call_llm(
                messages,
                base_url=LLM_BASE_URL,
                api_key=LLM_API_KEY,
                model=LLM_MODEL,
                temperature=LLM_TEMPERATURE,
                max_tokens=LLM_MAX_TOKENS,
            )
            results.append((req, raw))
            if args.delay > 0:
                time.sleep(args.delay)

    # 3. Parse and save
    print("[INFO] Saving results...")
    valid_count = 0
    passed_count = 0
    
    with open(args.out, "a", encoding="utf-8") as fout:
        for req, raw_text in results:
            buggy_code, chosen_bug_type, explanation = parse_response(raw_text)

            is_pass = (chosen_bug_type == "PASS" or not buggy_code.strip())
            
            if is_pass:
                chosen_bug_type = "PASS"
                passed_count += 1
                print(f"  → LLM passed on {req['bench']}/{req['fname']}: {explanation}")
            else:
                valid_count += 1

            row = {
                "benchmark": req["bench"],
                "source_file": req["fname"],
                "bug_type": chosen_bug_type,
                "original_code": req["source_code"],
                "buggy_code": "" if is_pass else buggy_code,
                "bug_explanation": explanation,
            }
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            
    print(f"[DONE] Processed {len(results)} files ({valid_count} bugs injected, {passed_count} passed) → wrotes entries to {args.out}")


if __name__ == "__main__":
    main()
