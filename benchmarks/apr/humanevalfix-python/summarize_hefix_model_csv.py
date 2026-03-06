#!/usr/bin/env python3
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path("hefix_generations")

# steer-tssb-qwencoder7b_Lband-0.15-6_a0p25_20251229-1529
COND_RE = re.compile(
    r"steer-(?P<vec_id>[^_]+)_"
    r"Lband-(?P<band>[0-9.]+)-(?P<width>[0-9]+)_"
    r"a(?P<strength>[0-9p]+)_"
    r"(?P<ts>[0-9\-]+)"
)

def parse_strength(s: str) -> float:
    return float(s.replace("p", "."))

def load_first_json_object(path: Path) -> dict:
    """
    Robustly parse a file that *starts* with a JSON object but may contain
    extra data afterward (e.g., duplicated JSON or appended logs).
    """
    txt = path.read_text(errors="replace").lstrip()

    decoder = json.JSONDecoder()
    obj, end = decoder.raw_decode(txt)  # parses the first JSON value
    # If you ever want the LAST json object instead, we can do that too.
    return obj

rows = []
bad = []

for model_dir in ROOT.iterdir():
    if not model_dir.is_dir():
        continue

    model_slug = model_dir.name

    for cond_dir in model_dir.iterdir():
        if not cond_dir.is_dir():
            continue
        if cond_dir.name == "baseline":
            continue

        metrics_path = cond_dir / "metrics.json"
        if not metrics_path.exists():
            continue

        m = COND_RE.match(cond_dir.name)
        if not m:
            bad.append((str(metrics_path), "condition_name_parse_failed"))
            continue

        try:
            metrics = load_first_json_object(metrics_path)
        except Exception as e:
            bad.append((str(metrics_path), f"json_parse_failed: {e}"))
            continue

        rows.append({
            "model": model_slug,
            "condition_dir": cond_dir.name,
            "condition": metrics.get("condition", cond_dir.name),
            "vec_id": m.group("vec_id"),
            "layer_band": float(m.group("band")),
            "band_width": int(m.group("width")),
            "strength": parse_strength(m.group("strength")),
            "timestamp": m.group("ts"),
            "n_tasks": metrics.get("n_tasks"),
            "total_completions": metrics.get("total_completions"),
            "invalid": metrics.get("invalid"),
            "invalid_rate": metrics.get("invalid_rate"),
            "pass@1": metrics.get("pass@1"),
            "pass@5": metrics.get("pass@5"),
            "pass@10": metrics.get("pass@10"),
        })

df = pd.DataFrame(rows)

out_dir = Path("summaries")
out_dir.mkdir(exist_ok=True)

# Write one CSV per model
for model, subdf in df.groupby("model", dropna=False):
    out_path = out_dir / f"{model}_summary.csv"
    subdf = subdf.sort_values(["layer_band", "strength", "timestamp"])
    subdf.to_csv(out_path, index=False)
    print(f"[INFO] Wrote {out_path} ({len(subdf)} rows)")

# Global CSV
df.sort_values(["model", "layer_band", "strength", "timestamp"]).to_csv(
    out_dir / "all_models_summary.csv", index=False
)
print(f"[INFO] Wrote {out_dir/'all_models_summary.csv'} ({len(df)} rows)")

# Report any problems
if bad:
    bad_path = out_dir / "bad_metrics_files.csv"
    pd.DataFrame(bad, columns=["path", "error"]).to_csv(bad_path, index=False)
    print(f"[WARN] Some files could not be parsed. See {bad_path} ({len(bad)} files).")

