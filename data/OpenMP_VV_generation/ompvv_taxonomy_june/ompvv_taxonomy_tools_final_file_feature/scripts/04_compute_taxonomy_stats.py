#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from io_utils import read_dataset

def write_counts(df: pd.DataFrame, cols: list[str], outpath: Path) -> None:
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return
    counts = (df.groupby(existing, dropna=False)
              .size()
              .reset_index(name="count")
              .sort_values("count", ascending=False))
    outpath.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(outpath, index=False)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    df = read_dataset(args.input)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    one_way = [
        "mutation_type", "omp_construct", "bug_omp_concept", "bug_type",
        "bug_family", "category", "source_dir", "openmp_version",
        "confidence", "passed_most_recent_test", "test_status", "_source_file"
    ]
    for col in one_way:
        write_counts(df, [col], outdir / f"counts_by_{col}.csv")

    cross_tabs = [
        ["mutation_type", "omp_construct"],
        ["mutation_type", "bug_omp_concept"],
        ["omp_construct", "bug_omp_concept"],
        ["mutation_type", "category"],
        ["omp_construct", "category"],
        ["openmp_version", "mutation_type"],
        ["source_dir", "mutation_type"],
        ["bug_family", "mutation_type"],
    ]
    for cols in cross_tabs:
        write_counts(df, cols, outdir / ("counts_by_" + "__".join(cols) + ".csv"))

    summary = {
        "num_rows": len(df),
        "num_unique_source_files": df["source_id"].nunique() if "source_id" in df else None,
        "num_unique_bug_types": df["bug_type"].nunique() if "bug_type" in df else None,
        "num_unique_mutation_types": df["mutation_type"].nunique() if "mutation_type" in df else None,
        "num_unique_omp_constructs": df["omp_construct"].nunique() if "omp_construct" in df else None,
        "num_unique_bug_omp_concepts": df["bug_omp_concept"].nunique() if "bug_omp_concept" in df else None,
    }
    pd.DataFrame([summary]).to_csv(outdir / "dataset_summary.csv", index=False)

    print(f"Wrote stats to {outdir}")
    for k, v in summary.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    main()
