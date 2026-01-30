import csv
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "meta" / "eval_results"

OUT_SUMMARY_CSV = RESULTS_DIR / "hejava_summary_all_models.csv"


def parse_filename(path: Path):
    """
    From a filename like:
      hejava_Qwen2.5-Coder-7B-Instruct_baseline_k1.csv
      hejava_Qwen2.5-Coder-7B-Instruct_steer-l24-25-26-27_a2_vqwencoder7b_k10.csv

    return (model_slug, condition, k)
    """
    name = path.name  # e.g., hejava_..._k1.csv
    # Strip prefix/suffix
    assert name.startswith("hejava_")
    base = name[len("hejava_"):]  # Qwen2.5-..._baseline_k1.csv
    base = base[:-len(".csv")]    # drop .csv

    # Extract k from the end (_k1)
    m = re.search(r"_k(\d+)$", base)
    if not m:
        raise ValueError(f"Cannot parse k from filename: {name}")
    k = int(m.group(1))

    # Remove the _kX at the end to isolate model + condition
    core = base[: m.start()]  # Qwen2.5-..._baseline

    # Split into model_slug and condition on the last underscore
    # model_slug may contain dashes, so split from the right
    parts = core.rsplit("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot split model/condition from: {core}")
    model_slug, condition = parts[0], parts[1]
    return model_slug, condition, k


def parse_summary_block(path: Path):
    """
    Read the SUMMARY block at the bottom of the CSV, which looks like:

    SUMMARY
    Total tasks,163
    Successful tasks,102
    Accuracy (pass@1),0.6258

    Returns (total_tasks, successes, accuracy_float).
    """
    total = None
    successes = None
    acc = None

    with path.open() as f:
        lines = [line.strip() for line in f if line.strip()]

    # Find the index of "SUMMARY"
    try:
        idx = lines.index("SUMMARY")
    except ValueError:
        # No summary block found
        raise RuntimeError(f"No SUMMARY block found in {path}")

    # Expect next 3 lines with key,value
    for line in lines[idx + 1 : idx + 4]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        key, val = parts
        if key.startswith("Total tasks"):
            total = int(val)
        elif key.startswith("Successful tasks"):
            successes = int(val)
        elif key.startswith("Accuracy (pass@"):
            acc = float(val)

    if total is None or successes is None or acc is None:
        raise RuntimeError(f"Incomplete SUMMARY block in {path}")

    return total, successes, acc


def main():
    print(f"[INFO] Results dir: {RESULTS_DIR}")
    files = sorted(RESULTS_DIR.glob("hejava_*.csv"))
    if not files:
        print("[WARN] No hejava_*.csv files found.")
        return

    rows = []
    for path in files:
        try:
            model_slug, condition, k = parse_filename(path)
            total, successes, acc = parse_summary_block(path)
        except Exception as e:
            print(f"[WARN] Skipping {path.name}: {e}")
            continue

        variant = "baseline" if condition == "baseline" else "steered"

        rows.append(
            {
                "model_slug": model_slug,
                "condition": condition,
                "variant": variant,  # baseline / steered
                "k": k,
                "total_tasks": total,
                "successful_tasks": successes,
                "accuracy": acc,  # already in [0,1]
            }
        )

    # Write aggregated CSV
    print(f"[INFO] Writing summary to: {OUT_SUMMARY_CSV}")
    with OUT_SUMMARY_CSV.open("w", newline="") as f:
        fieldnames = [
            "model_slug",
            "condition",
            "variant",
            "k",
            "total_tasks",
            "successful_tasks",
            "accuracy",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote {len(rows)} rows.")
    print("[INFO] Example rows:")
    for r in rows[:5]:
        print("  ", r)


if __name__ == "__main__":
    main()

