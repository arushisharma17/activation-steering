from pathlib import Path
import csv

# meta/scripts -> meta -> project root
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]

BUGGY_DIR = PROJECT_ROOT / "src" / "main" / "java" / "humaneval" / "buggy"
TEST_DIR  = PROJECT_ROOT / "src" / "test" / "java" / "humaneval"
OUT_CSV   = PROJECT_ROOT / "meta" / "java_tasks.csv"


def main():
    print(f"[INFO] Project root: {PROJECT_ROOT}")
    print(f"[INFO] Buggy dir   : {BUGGY_DIR}")
    print(f"[INFO] Test dir    : {TEST_DIR}")
    print(f"[INFO] Output CSV  : {OUT_CSV}")

    if not BUGGY_DIR.exists():
        raise SystemExit(f"[ERROR] Buggy dir not found: {BUGGY_DIR}")
    if not TEST_DIR.exists():
        raise SystemExit(f"[ERROR] Test dir not found: {TEST_DIR}")

    buggy_files = sorted(BUGGY_DIR.glob("*.java"))
    if not buggy_files:
        raise SystemExit(f"[ERROR] No .java files found in {BUGGY_DIR}")

    rows = []
    skipped = 0

    for idx, bf in enumerate(buggy_files):
        java_name = bf.stem  # e.g., STRING_TO_MD5 from STRING_TO_MD5.java

        # Expected test file name and class
        test_name = f"TEST_{java_name}.java"
        test_file = TEST_DIR / test_name
        test_class = f"humaneval.TEST_{java_name}"

        if not test_file.exists():
            print(f"[WARN] No test file for {java_name}: expected {test_file}")
            skipped += 1
            continue

        task_id = f"HEJ_{idx:03d}"  # HEJ_000, HEJ_001, ...

        buggy_path = str(bf.relative_to(PROJECT_ROOT))  # e.g. src/main/java/humaneval/buggy/STRING_TO_MD5.java

        rows.append({
            "task_id": task_id,
            "java_name": java_name,
            "buggy_path": buggy_path,
            "test_class": test_class,
        })

    if not rows:
        raise SystemExit("[ERROR] No tasks discovered; CSV would be empty.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f_out:
        writer = csv.DictWriter(
            f_out,
            fieldnames=["task_id", "java_name", "buggy_path", "test_class"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Wrote {len(rows)} tasks to {OUT_CSV}")
    if skipped:
        print(f"[INFO] Skipped {skipped} buggy file(s) with no matching TEST_*.java")


if __name__ == "__main__":
    main()
