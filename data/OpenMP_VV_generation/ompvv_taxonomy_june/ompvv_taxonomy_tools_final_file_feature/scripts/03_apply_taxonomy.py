#!/usr/bin/env python3
from __future__ import annotations
import argparse
import pandas as pd
from io_utils import read_dataset, write_dataset
from taxonomy_rules import infer_source_dir, infer_openmp_version

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--output-json", required=False)
    parser.add_argument("--output-csv", required=False)
    parser.add_argument("--fail-on-unmapped", action="store_true")
    args = parser.parse_args()

    df = read_dataset(args.input)
    mapping = pd.read_csv(args.mapping)
    needed = ["bug_type", "mutation_type", "omp_construct", "bug_omp_concept"]
    missing = [c for c in needed if c not in mapping.columns]
    if missing:
        raise ValueError(f"Mapping missing columns: {missing}")

    mapping_small = mapping[needed].drop_duplicates("bug_type")
    out = df.merge(mapping_small, on="bug_type", how="left")
    out["source_dir"] = out["source_id"].apply(infer_source_dir)
    out["openmp_version"] = out["source_id"].apply(infer_openmp_version)

    unmapped = out[out["mutation_type"].isna()]["bug_type"].dropna().unique()
    if len(unmapped) > 0:
        msg = f"Unmapped bug types: {sorted(unmapped)}"
        if args.fail_on_unmapped:
            raise ValueError(msg)
        print("WARNING:", msg)
        out["mutation_type"] = out["mutation_type"].fillna("NEEDS_REVIEW")
        out["omp_construct"] = out["omp_construct"].fillna("unknown")
        out["bug_omp_concept"] = out["bug_omp_concept"].fillna("unknown")

    if args.output_json:
        write_dataset(out, args.output_json)
        print(f"Wrote {args.output_json}")
    if args.output_csv:
        write_dataset(out, args.output_csv)
        print(f"Wrote {args.output_csv}")
    if not args.output_json and not args.output_csv:
        raise ValueError("Provide --output-json and/or --output-csv")

    print(f"Rows: {len(out)}")
    print(f"Unique bug types: {out['bug_type'].nunique()}")
    print(f"Unique mutation types: {out['mutation_type'].nunique()}")
    print(f"Unique omp_construct: {out['omp_construct'].nunique()}")
    print(f"Unique bug_omp_concept: {out['bug_omp_concept'].nunique()}")

if __name__ == "__main__":
    main()
