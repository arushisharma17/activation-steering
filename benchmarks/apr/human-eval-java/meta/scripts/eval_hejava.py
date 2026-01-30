import argparse
import csv
import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Paths (similar to other scripts)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

GENERATIONS_ROOT = PROJECT_ROOT / "hej_generations"
TASKS_CSV = PROJECT_ROOT / "meta" / "java_tasks.csv"
BUGGY_ORIGINAL_DIR = PROJECT_ROOT / "data" / "buggy_original"


def get_maven_repo() -> Path:
    """
    Determine where Maven's local repo should live on the cluster.
    Prefer the MAVEN_REPO env var if set; otherwise default to PROJECT_ROOT/maven-repo.
    """
    env_repo = os.environ.get("MAVEN_REPO")
    if env_repo:
        return Path(env_repo)
    return PROJECT_ROOT / "maven-repo"


def parse_sample_index(path: Path) -> int:
    """
    Given a file like HEJ_000_STRING_TO_MD5_s3.java, return 3.
    """
    stem = path.stem  # e.g. HEJ_000_STRING_TO_MD5_s3
    if "_s" not in stem:
        return 0
    try:
        return int(stem.split("_s", 1)[1])
    except Exception:
        return 0


def run_mvn_test(test_class: str, maven_repo: Path) -> bool:
    """
    Run Maven tests for a single test class, using a custom local repo directory.
    Returns True if the test passes (exit code 0).
    """
    cmd = [
        "mvn",
        "-q",
        f"-Dmaven.repo.local={maven_repo}",
        f"-Dtest={test_class}",
        "test",
    ]
    print(f"[INFO] Running: {' '.join(cmd)}")
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
    )
    ok = (proc.returncode == 0)
    print(f"[INFO] Maven test for {test_class} -> {'PASS' if ok else 'FAIL'} (rc={proc.returncode})")
    return ok


def evaluate_task(
    task_row: Dict[str, str],
    condition_root: Path,
    max_k: int,
    maven_repo: Path,
) -> Tuple[bool, Optional[int]]:
    """
    Evaluate one HumanEval-Java task.

    - task_row: row from java_tasks.csv
    - condition_root: hej_generations/<model_slug>/<condition>
    - max_k: maximum number of samples per task to consider

    Returns (success, first_success_index) where:
      - success: True if any sample passes within k
      - first_success_index: the sample index (0-based) that first passed, or None
    """
    task_id = task_row["task_id"]
    java_name = task_row["java_name"]
    buggy_path = task_row["buggy_path"]
    test_class = task_row["test_class"]

    print(f"\n[INFO] === Task {task_id} ({java_name}) ===")

    # Where the original buggy file lives in src/
    buggy_file = PROJECT_ROOT / buggy_path
    original_file = BUGGY_ORIGINAL_DIR / f"{java_name}.java"

    if not original_file.exists():
        print(f"[WARN] Original buggy snapshot missing: {original_file}")
        return False, None
    if not buggy_file.exists():
        print(f"[WARN] Buggy file path from CSV does not exist: {buggy_file}")
        return False, None

    # Collect all generated samples for this task
    pattern = f"{task_id}_{java_name}_s*.java"
    gen_files = sorted(
        condition_root.glob(pattern),
        key=parse_sample_index,
    )

    if not gen_files:
        print(f"[WARN] No generation files found matching {pattern} under {condition_root}")
        return False, None

    # Limit to first max_k samples
    gen_files = gen_files[:max_k]

    print(f"[INFO] Found {len(gen_files)} sample(s) for this task (max_k={max_k}).")

    success = False
    first_success_idx: Optional[int] = None

    try:
        for idx, gen_path in enumerate(gen_files):
            print(f"[INFO] Trying sample s{parse_sample_index(gen_path)}: {gen_path.name}")

            # Overwrite buggy file with generated fixed code
            shutil.copy(gen_path, buggy_file)

            # Run the test using the custom Maven repo
            if run_mvn_test(test_class, maven_repo=maven_repo):
                success = True
                first_success_idx = idx
                print(f"[INFO] Task {task_id} SUCCESS at sample index {idx}")
                break
            else:
                print(f"[INFO] Sample {gen_path.name} failed.")
    finally:
        # Always restore the original buggy file
        print(f"[INFO] Restoring original buggy file for {java_name}")
        shutil.copy(original_file, buggy_file)

    return success, first_success_idx


def main():
    ap = argparse.ArgumentParser(description="Evaluate HumanEval-Java generations with Maven tests.")
    ap.add_argument(
        "--model_slug",
        required=True,
        help="Model slug used in hej_generations/<model_slug>/..., e.g., 'Qwen2.5-Coder-7B-Instruct'.",
    )
    ap.add_argument(
        "--condition",
        default="baseline",
        help="Condition directory under the model slug (default: 'baseline').",
    )
    ap.add_argument(
        "--max_k",
        type=int,
        default=1,
        help="Max number of samples per task to consider (pass@k).",
    )

    args = ap.parse_args()

    condition_root = GENERATIONS_ROOT / args.model_slug / args.condition
    maven_repo = get_maven_repo()

    print(f"[INFO] Project root     : {PROJECT_ROOT}")
    print(f"[INFO] Tasks CSV        : {TASKS_CSV}")
    print(f"[INFO] Generations root : {condition_root}")
    print(f"[INFO] max_k            : {args.max_k}")
    print(f"[INFO] Maven repo local : {maven_repo}")

    if not condition_root.exists():
        raise SystemExit(f"[ERROR] Condition directory not found: {condition_root}")
    if not TASKS_CSV.exists():
        raise SystemExit(f"[ERROR] Tasks CSV not found: {TASKS_CSV}")
    if not BUGGY_ORIGINAL_DIR.exists():
        raise SystemExit(f"[ERROR] Missing {BUGGY_ORIGINAL_DIR}; cannot restore originals.")

    # Ensure Maven repo dir exists
    maven_repo.mkdir(parents=True, exist_ok=True)

    # Load tasks
    with open(TASKS_CSV) as f_in:
        tasks = list(csv.DictReader(f_in))

    print(f"[INFO] Loaded {len(tasks)} tasks from {TASKS_CSV}")

    total = 0
    success_count = 0
    per_task_results: List[Dict[str, object]] = []

    for trow in tasks:
        total += 1
        ok, first_idx = evaluate_task(
            task_row=trow,
            condition_root=condition_root,
            max_k=args.max_k,
            maven_repo=maven_repo,
        )
        if ok:
            success_count += 1
        per_task_results.append(
            {
                "task_id": trow["task_id"],
                "java_name": trow["java_name"],
                "success": ok,
                "first_success_sample_index": first_idx,
            }
        )

    accuracy = success_count / total if total > 0 else 0.0
    print("\n[INFO] ===== SUMMARY =====")
    print(f"[INFO] Total tasks        : {total}")
    print(f"[INFO] Successful tasks   : {success_count}")
    print(f"[INFO] Accuracy (pass@{args.max_k}) : {accuracy * 100:.2f}%")

    # Write per-task results CSV
    out_dir = PROJECT_ROOT / "meta" / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"hejava_{args.model_slug}_{args.condition}_k{args.max_k}.csv"

    print(f"[INFO] Writing per-task results to {out_csv}")
    # First: write header + per-task rows
    with open(out_csv, "w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["task_id", "java_name", "success", "first_success_sample_index"],
        )
        writer.writeheader()
        writer.writerows(per_task_results)

        # Then: blank line + summary block
        f_out.write("\n")
        summary_writer = csv.writer(f_out)
        summary_writer.writerow(["SUMMARY"])
        summary_writer.writerow(["Total tasks", total])
        summary_writer.writerow(["Successful tasks", success_count])
        summary_writer.writerow([f"Accuracy (pass@{args.max_k})", f"{accuracy:.4f}"])

    print(f"[INFO] Wrote per-task results to {out_csv}")

if __name__ == "__main__":
    main()

