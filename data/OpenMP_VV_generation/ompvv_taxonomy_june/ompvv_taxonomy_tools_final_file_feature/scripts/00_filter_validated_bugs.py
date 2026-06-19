#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from io_utils import read_dataset, write_dataset

def as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().map({"true": True, "false": False})

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-valid", required=True)
    parser.add_argument("--output-rejected", required=True)
    parser.add_argument("--summary", required=True)
    args = parser.parse_args()

    df = read_dataset(args.input)
    if "passed_most_recent_test" not in df.columns:
        raise ValueError("Dataset must contain passed_most_recent_test column.")

    passed = as_bool_series(df["passed_most_recent_test"])
    valid = df[passed == False].copy()
    rejected = df[passed == True].copy()

    write_dataset(valid, args.output_valid)
    write_dataset(rejected, args.output_rejected)

    summary = pd.DataFrame([
        {"subset": "validated_failing_bugs", "filter": "passed_most_recent_test == false", "count": len(valid)},
        {"subset": "rejected_passing_mutations", "filter": "passed_most_recent_test == true", "count": len(rejected)},
        {"subset": "total", "filter": "all rows", "count": len(df)},
    ])
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)

    print(f"Wrote {args.output_valid}: {len(valid)} rows")
    print(f"Wrote {args.output_rejected}: {len(rejected)} rows")
    print(f"Wrote {args.summary}")

if __name__ == "__main__":
    main()
