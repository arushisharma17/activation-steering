# summarize_hefix_results.py
import os
import glob
import json
import datetime
import pandas as pd


def main():
    gen_root = "hefix_generations"
    metric_files = glob.glob(os.path.join(gen_root, "*", "*", "metrics.json"))
    if not metric_files:
        print("[WARN] No metrics.json files found under hefix_generations/")
        return

    rows = []
    for mf in metric_files:
        with open(mf, "r") as f:
            m = json.load(f)
        m["path"] = mf
        rows.append(m)

    df = pd.DataFrame(rows)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(gen_root, f"hefix_summary_{ts}.csv")
    df.to_csv(out, index=False)
    print(f"[INFO] Wrote {out}")


if __name__ == "__main__":
    main()

