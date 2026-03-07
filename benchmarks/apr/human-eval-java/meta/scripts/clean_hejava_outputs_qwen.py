#!/usr/bin/env python3
import argparse
from pathlib import Path

def clean_java_source(text: str) -> str:
    """
    Clean an already-generated Java source file:
    - Remove leading `java`, ```java, or language tags
    - Remove trailing ``` fences
    - Strip junk before/after class definition
    """

    fence = "`" * 3

    # If the file contains multiple fenced blocks, keep the best block
    if fence in text:
        parts = [p.strip() for p in text.split(fence) if p.strip()]
        # Prefer a segment containing both package + class
        for p in reversed(parts):
            if "package " in p and "class " in p:
                text = p
                break
        else:
            # fallback: longest part
            text = max(parts, key=len)

    # Split into lines
    lines = text.splitlines()

    # Remove leading garbage like "java", "Java", "```java", "language: java"
    bad_prefixes = {"java", "Java", "```java", "language: java"}
    while lines and lines[0].strip().lower() in bad_prefixes:
        lines.pop(0)

    # Remove any closing ``` fence in the middle
    cleaned = []
    for line in lines:
        if line.strip().startswith("```"):
            break
        cleaned.append(line)

    return "\n".join(cleaned).strip()


def main():
    ap = argparse.ArgumentParser(description="Clean already generated HEJava .java files.")
    ap.add_argument(
        "--root",
        required=True,
        help="Path to a directory containing generated .java files "
             "(e.g., hej_generations/Qwen2.5-7B-Instruct/steer-l24-25-26-27_a2_vqwen7b-general)",
    )
    args = ap.parse_args()

    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"[ERROR] Path does not exist: {root}")

    java_files = list(root.rglob("*.java"))
    print(f"[INFO] Found {len(java_files)} .java files under {root}")

    changed = 0

    for jf in java_files:
        orig = jf.read_text()
        cleaned = clean_java_source(orig)

        if cleaned != orig:
            jf.write_text(cleaned)
            changed += 1
            print(f"[CLEAN] {jf}")

    print(f"\n[INFO] Cleanup complete. Modified {changed} file(s).")


if __name__ == "__main__":
    main()

