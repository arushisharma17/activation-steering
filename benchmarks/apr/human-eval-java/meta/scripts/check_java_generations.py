#!/usr/bin/env python
import argparse
import csv
import sys
from pathlib import Path


def detect_issues(text: str) -> list[str]:
    issues = []
    raw = text
    stripped = raw.strip()
    lower = stripped.lower()

    # 1) trivial / structural issues
    if not stripped:
        issues.append("EMPTY_FILE")

    if len(stripped) < 30:
        issues.append("SUSPICIOUSLY_SHORT")

    # 2) markdown / language tags / artifacts
    if "```" in raw:
        issues.append("HAS_MARKDOWN_FENCES")

    if lower.startswith("java\n") or lower.startswith("java "):
        issues.append("LEADING_LANGUAGE_TAG")

    if "<corrected" in lower or "corrected>" in lower:
        issues.append("HAS_CORRECTED_TAG")

    if "here is the corrected" in lower or "corrected java class" in lower:
        issues.append("HAS_EXPLANATORY_TEXT")

    # 3) Java-specific structure
    if "package " not in raw:
        issues.append("MISSING_PACKAGE")

    if "class " not in raw:
        issues.append("MISSING_CLASS")

    # 4) Garbage before package (non-comment text)
    pkg_idx = raw.find("package ")
    if pkg_idx > 0:
        prefix = raw[:pkg_idx].strip()
        if prefix:
            # allow pure comments or blank lines only
            # crude check: if there is any line not starting with a comment marker
            bad_prefix = False
            for line in prefix.splitlines():
                l = line.strip()
                if not l:
                    continue
                if not (l.startswith("//") or l.startswith("/*") or l.startswith("*") or l.startswith("*/")):
                    bad_prefix = True
                    break
            if bad_prefix:
                issues.append("GARBAGE_BEFORE_PACKAGE")

    # 5) Brace balance (very rough)
    open_braces = raw.count("{")
    close_braces = raw.count("}")
    if open_braces != close_braces:
        issues.append(f"UNBALANCED_BRACES({open_braces}:{close_braces})")

    return issues


def parse_task_meta(path: Path):
    """
    Try to derive (task_id, java_name, sample_idx) from filename like:
      HEJ_000_ADD_s0.java
    """
    stem = path.stem  # e.g. HEJ_000_ADD_s0
    parts = stem.split("_")
    if len(parts) < 3:
        return stem, "", ""

    task_id = "_".join(parts[0:2])  # HEJ_000
    # everything until last part is java_name
    java_name = "_".join(parts[2:-1]) if len(parts) > 3 else parts[2]
    s_part = parts[-1]  # s0
    sample_idx = ""
    if s_part.startswith("s"):
        sample_idx = s_part[1:]
    return task_id, java_name, sample_idx


def main():
    ap = argparse.ArgumentParser(
        description="Scan HumanEval-Java generations for formatting issues."
    )
    ap.add_argument(
        "--dir",
        required=True,
        help="Directory containing .java generations (e.g. hej_generations/CodeLlama-7b-Instruct-hf/baseline)",
    )
    ap.add_argument(
        "--csv",
        default=None,
        help="Optional path to write a CSV summary of issues.",
    )
    args = ap.parse_args()

    root = Path(args.dir).resolve()
    if not root.exists() or not root.is_dir():
        print(f"[ERROR] Directory not found or not a dir: {root}", file=sys.stderr)
        sys.exit(1)

    java_files = sorted(root.rglob("*.java"))
    if not java_files:
        print(f"[WARN] No .java files found under {root}")
        sys.exit(0)

    print(f"[INFO] Scanning {len(java_files)} .java files under {root}")

    rows = []
    total_with_issues = 0

    for jf in java_files:
        text = jf.read_text(encoding="utf-8", errors="replace")
        issues = detect_issues(text)

        task_id, java_name, sample_idx = parse_task_meta(jf)

        if issues:
            total_with_issues += 1
            issue_str = ";".join(issues)
            rows.append(
                {
                    "file": str(jf.relative_to(root)),
                    "task_id": task_id,
                    "java_name": java_name,
                    "sample_idx": sample_idx,
                    "num_issues": len(issues),
                    "issues": issue_str,
                }
            )

    # Pretty print a short report to stdout
    print()
    print(f"[INFO] Files with issues: {total_with_issues} / {len(java_files)}")
    if rows:
        print("\n[INFO] Sample problematic files:")
        for r in rows[:20]:
            print(
                f"  - {r['file']}  [task={r['task_id']}, s={r['sample_idx']}]  issues={r['issues']}"
            )

    # Optional CSV export
    if args.csv:
        csv_path = Path(args.csv).resolve()
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file",
                    "task_id",
                    "java_name",
                    "sample_idx",
                    "num_issues",
                    "issues",
                ],
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        print(f"\n[INFO] CSV summary written to: {csv_path}")

    print("\n[INFO] Done.")


if __name__ == "__main__":
    main()

