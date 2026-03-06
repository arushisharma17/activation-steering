# eval_hefix.py
from __future__ import annotations

import argparse
import json
import os
import re
import ast
from collections import defaultdict
from typing import Dict, Any, List

from human_eval.data import stream_jsonl, write_jsonl
from human_eval.execution import check_correctness

from humanevalfix_data import read_hefix_problems


def extract_code(text: str) -> str:
    """
    Extract runnable Python code from a model completion.

    - If fenced blocks exist, use the *first* block that contains a def/class (or just first block).
    - Preserve top-level imports / future imports that appear before the first def/class.
    - Keep everything from the first def/class onward.
    """
    if text is None:
        return ""

    text = text.strip()
    if not text:
        return ""

    # 1) If there are fenced blocks, pick the best one
    if "```" in text:
        parts = text.split("```")
        # parts looks like: [outside, inside1, outside2, inside2, ...]
        fenced_candidates: List[str] = []
        for i in range(1, len(parts), 2):
            p = parts[i].strip()
            if not p:
                continue
            # drop optional language tag line
            lines = p.splitlines()
            if lines and lines[0].strip().lower() in {"python", "py"}:
                p = "\n".join(lines[1:]).strip()
            if p:
                fenced_candidates.append(p)

        # Prefer a fenced block that contains a def/class (allow indentation)
        for p in fenced_candidates:
            if re.search(r"^\s*(def|class)\s+", p, flags=re.MULTILINE):
                text = p
                break
        else:
            # Otherwise use the first non-empty fenced block
            if fenced_candidates:
                text = fenced_candidates[0]

    lines = text.splitlines()

    # 2) Preserve a preamble of imports / future imports / encoding line
    preamble: List[str] = []
    i_def = None

    for i, line in enumerate(lines):
        s = line.strip()

        # keep shebang / encoding lines
        if i == 0 and s.startswith("#!"):
            preamble.append(line)
            continue
        if re.match(r"^#.*coding[:=]\s*[-\w.]+", s):
            preamble.append(line)
            continue

        if not s:
            # keep blank lines in preamble (harmless)
            preamble.append(line)
            continue

        # keep import statements in the preamble (allow indentation for future import)
        if s.startswith("from __future__ import ") or s.startswith("import ") or s.startswith("from "):
            preamble.append(line)
            continue

        # first code definition starts here (allow indentation)
        if re.match(r"^\s*(def|class)\s+", line):
            i_def = i
            break

        # Otherwise ignore prose/etc and keep scanning for first def/class
        continue

    # 3) Keep from first def/class onward (or fallback to whole text if none found)
    if i_def is None:
        return text.strip()

    body = "\n".join(lines[i_def:]).strip()
    pre = "\n".join(preamble).strip()

    if pre:
        return (pre + "\n\n" + body).strip()
    return body


def compute_pass_at_k(pass_list: List[bool], k: int) -> float:
    return 1.0 if any(pass_list[:k]) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_slug", required=True)
    ap.add_argument("--condition", required=True)
    ap.add_argument("--gen_root", default="hefix_generations")
    ap.add_argument("--timeout", type=float, default=5.0)
    ap.add_argument("--k_list", default="1,5,10")
    args = ap.parse_args()

    ks = [int(x) for x in args.k_list.split(",") if x.strip()]

    run_dir = os.path.join(args.gen_root, args.model_slug, args.condition)
    sample_path = os.path.join(run_dir, "samples.jsonl")
    results_path = os.path.join(run_dir, "results.jsonl")
    metrics_path = os.path.join(run_dir, "metrics.json")

    if not os.path.exists(sample_path):
        raise FileNotFoundError(f"Missing samples file: {sample_path}")

    problems = read_hefix_problems(split="test")

    # Store completions per task; if "completion" is empty, fall back to "raw_completion"
    by_task: Dict[str, List[str]] = defaultdict(list)
    for row in stream_jsonl(sample_path):
        comp = row.get("completion", "")
        if not comp:
            comp = row.get("raw_completion", "")
        by_task[str(row["task_id"])].append(comp or "")

    results_rows: List[Dict[str, Any]] = []
    task_pass_lists: Dict[str, List[bool]] = {}

    total = 0
    invalid = 0

    for task_id, comps in by_task.items():
        prob = problems.get(task_id)
        if prob is None:
            continue

        entry = prob["entry_point"]

        he_problem = {
            "task_id": task_id,
            "prompt": "",  # completion should include full corrected code
            "test": prob["test"],
            "entry_point": entry,
        }

        pass_list: List[bool] = []
        for i, comp in enumerate(comps):
            total += 1
            code = extract_code(comp)

            # Treat empty outputs as invalid
            if not code.strip():
                invalid += 1
                pass_list.append(False)
                results_rows.append(
                    {
                        "task_id": task_id,
                        "completion_id": i,
                        "passed": False,
                        "result": "EMPTY_COMPLETION",
                        "exception": None,
                    }
                )
                continue

            # Require the correct entrypoint to appear in the extracted code
            if not re.search(
                rf"^\s*def\s+{re.escape(entry)}\s*\(",
                code,
                flags=re.MULTILINE,
            ):
                invalid += 1
                pass_list.append(False)
                results_rows.append(
                    {
                        "task_id": task_id,
                        "completion_id": i,
                        "passed": False,
                        "result": f"MISSING_ENTRYPOINT:{entry}",
                        "exception": None,
                    }
                )
                continue

            # Optional: syntax check before sandbox execution (avoids timeouts on junk)
            try:
                ast.parse(code)
            except SyntaxError as e:
                invalid += 1
                pass_list.append(False)
                results_rows.append(
                    {
                        "task_id": task_id,
                        "completion_id": i,
                        "passed": False,
                        "result": "SYNTAX_ERROR",
                        "exception": f"{e.msg} (line {e.lineno})",
                    }
                )
                continue

            try:
                res = check_correctness(
                    he_problem,
                    code,
                    timeout=args.timeout,
                    completion_id=i,
                )
                passed = bool(res.get("passed", False))
                pass_list.append(passed)

                results_rows.append(
                    {
                        "task_id": task_id,
                        "completion_id": i,
                        "passed": passed,
                        "result": res.get("result", None),
                        "exception": res.get("exception", None),
                    }
                )
            except Exception as e:
                invalid += 1
                pass_list.append(False)
                results_rows.append(
                    {
                        "task_id": task_id,
                        "completion_id": i,
                        "passed": False,
                        "result": "ERROR",
                        "exception": repr(e),
                    }
                )

        task_pass_lists[task_id] = pass_list

    n_tasks = len(task_pass_lists)
    pass_at = {f"pass@{k}": 0.0 for k in ks}

    if n_tasks > 0:
        for k in ks:
            s = 0.0
            for _, pl in task_pass_lists.items():
                s += compute_pass_at_k(pl, k)
            pass_at[f"pass@{k}"] = s / n_tasks

    metrics = {
        "model_slug": args.model_slug,
        "condition": args.condition,
        "n_tasks": n_tasks,
        "total_completions": total,
        "invalid": invalid,
        "invalid_rate": (invalid / total) if total else 0.0,
        **pass_at,
    }

    write_jsonl(results_path, results_rows)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"[INFO] Wrote {results_path}")
    print(f"[INFO] Wrote {metrics_path}")
    print("[INFO] Metrics:", metrics)


if __name__ == "__main__":
    main()

