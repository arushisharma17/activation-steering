from __future__ import annotations
from pathlib import Path
import json
import pandas as pd

def read_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.DataFrame(json.loads(path.read_text(encoding="utf-8")))
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input format: {path}")

def write_dataset(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
    elif suffix == ".json":
        path.write_text(df.to_json(orient="records", indent=2), encoding="utf-8")
    elif suffix in {".jsonl", ".ndjson"}:
        df.to_json(path, orient="records", lines=True)
    else:
        raise ValueError(f"Unsupported output format: {path}")
