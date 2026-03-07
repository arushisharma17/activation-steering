#!/usr/bin/env python

import argparse
from pathlib import Path

def cleanup_java_source(text: str) -> str:
    """
    Heuristics to turn a messy model output into a single valid Java class:

    - Remove leading junk before the first 'package ' line.
    - Remove '<corrected>' tags or similar markers.
    - Trim everything after the last balanced '}' that closes the class.
    """
    if not text.strip():
        return text

    lines = text.splitlines()

    # 1) Strip leading "<corrected>" or similar non-Java junk
    cleaned = []
    for ln in lines:
        stripped = ln.strip()
        # Drop obvious non-code markers
        if stripped.lower().startswith("<corrected"):
            continue
        cleaned.append(ln)
    lines = cleaned

    # 2) Start from first 'package ' line if it exists
    start_idx = 0
    for i, ln in enumerate(lines):
        if ln.strip().startswith("package "):
            start_idx = i
            break
    lines = lines[start_idx:]

    # 3) Find the end of the main Java class by tracking braces
    brace_balance = 0
    class_seen = False
    end_idx = len(lines)

    for i, ln in enumerate(lines):
        if "class " in ln:
            class_seen = True

        # Update brace balance
        brace_balance += ln.count("{")
        brace_balance -= ln.count("}")

        # Once we've seen a class and balance returns to 0, assume class is closed
        if class_seen and brace_balance == 0:
            end_idx = i + 1  # include this line
            # Don't break; sometimes extra '}'s appear, but we want the first full class.
            break

    core = lines[:end_idx]

    # 4) Optionally, drop trailing instruction-like lines if they survived
    filtered = []
    for ln in core:
        stripped = ln.strip()
        if stripped.startswith("Note that the output must be"):
            continue
        if stripped.startswith("You must ONLY output"):
            continue
        if stripped.startswith("You may ONLY modify"):
            continue
        filtered.append(ln)

    return "\n".join(filtered).rstrip() + "\n"


def process_dir(root: Path, dry_run: bool = False):
    java_files = sorted(root.glob("*.java"))
    print(f"[INFO] Found {len(java_files)} .java files under {root}")

    num_changed = 0
    for fp in java_files:
        original = fp.read_text()
        cleaned = cleanup_java_source(original)
        if cleaned != original:
            num_changed += 1
            print(f"[MODIFIED] {fp}")
            if not dry_run:
                fp.write_text(cleaned)

    print(f"[INFO] Done. Modified {num_changed} file(s).")


def main():
    ap = argparse.ArgumentParser(description="Clean up HumanEval-Java Codellama generations.")
    ap.add_argument(
        "--dir",
        required=True,
        help="Directory with .java generations (e.g., hej_generations/CodeLlama-7b-Instruct-hf/baseline)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not overwrite files, just report what would change.",
    )
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] Directory not found: {root}")

    process_dir(root, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

