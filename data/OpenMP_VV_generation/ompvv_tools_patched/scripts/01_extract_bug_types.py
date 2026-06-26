#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from io_utils import read_dataset
from taxonomy_rules import classify_row

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = read_dataset(args.input)
    rows = []
    for bug_type, group in df.groupby("bug_type", dropna=False):
        sample = group.iloc[0].to_dict()
        inferred = classify_row(sample)
        rows.append({
            "bug_type": bug_type,
            "count": len(group),
            "example_source_id": sample.get("source_id", ""),
            "example_category": sample.get("category", ""),
            "example_bug_family": sample.get("bug_family", ""),
            "example_original_statement": sample.get("original_statement", ""),
            "example_buggy_statement": sample.get("buggy_statement", ""),
            "suggested_mutation_type": inferred["mutation_type"],
            "suggested_omp_construct": inferred["omp_construct"],
            "suggested_bug_omp_concept": inferred["bug_omp_concept"],
            "needs_review": inferred["needs_review"],
        })

    out = pd.DataFrame(rows).sort_values(["needs_review", "suggested_mutation_type", "bug_type"])
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {args.output}")
    print(f"Examples: {len(df)}")
    print(f"Unique bug types: {out['bug_type'].nunique()}")
    print(f"Needs review: {(out['needs_review'].astype(str).str.upper() == 'TRUE').sum()}")

if __name__ == "__main__":
    main()
