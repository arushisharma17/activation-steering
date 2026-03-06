# humanevalfix_data.py
from __future__ import annotations
from typing import Dict, Any
from datasets import load_dataset


def read_hefix_problems(split: str = "test") -> Dict[str, Dict[str, Any]]:
    """
    HumanEvalFix-Python via bigcode/humanevalpack (config='python').
    Returns a dict keyed by task_id with:
      prompt, buggy_solution, test, entry_point, canonical_solution (optional).
    """
    ds = load_dataset("bigcode/humanevalpack", "python")[split]

    # 🔹 SMOKE TEST: limit dataset size
    #if split == "test":
    #    ds = ds.select(range(2))   # load only first 2 problems


    problems: Dict[str, Dict[str, Any]] = {}

    for ex in ds:
        tid = ex.get("task_id") or ex.get("id") or ex.get("name")
        if tid is None:
            tid = f"hefix_python_{len(problems)}"
        tid = str(tid)

        problems[tid] = {
            "task_id": tid,
            "prompt": ex["prompt"],
            "buggy_solution": ex["buggy_solution"],
            "canonical_solution": ex.get("canonical_solution", ""),
            "test": ex["test"],
            "entry_point": ex["entry_point"],
        }

    return problems


def build_prompt_generic(task_row: Dict[str, Any]) -> str:
    """
    Prompt mirroring your Java structure: invariants + <buggy> tag + code-only output.
    """
    entry = task_row["entry_point"]
    buggy_code = task_row["prompt"] + task_row["buggy_solution"]

    return (
        "You are an expert Python developer. You must FIX the following Python function so that it passes "
        "all of its existing unit tests.\n\n"
        "CRITICAL STRUCTURE RULES (do not break these):\n"
        f"- Keep the EXACT SAME function name (entry point): {entry}\n"
        "- Keep the EXACT SAME function signature (parameters).\n"
        "- Keep ALL required imports if present in the buggy code.\n"
        "- Only change the internal logic needed so the tests pass.\n"
        "- Do NOT add extra helper functions or unrelated code.\n"
        "- Do NOT include explanations, comments, or markdown.\n"
        "- Output ONLY the corrected Python code, as valid Python source code. Include any necessary imports (e.g., from typing import List). Do not use markdown fences.\n\n"
        "Here is the current buggy implementation, between <buggy> tags:\n\n"
        "<buggy>\n"
        f"{buggy_code}\n"
        "</buggy>\n\n"
        "Now write ONLY the corrected Python code below this line:\n\n"
    )

