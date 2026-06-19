#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t = pd.read_csv(args.template)
    out = pd.DataFrame({
        "bug_type": t["bug_type"],
        "mutation_type": t["suggested_mutation_type"],
        "omp_construct": t["suggested_omp_construct"],
        "bug_omp_concept": t["suggested_bug_omp_concept"],
        "needs_review": t["needs_review"],
        "count": t["count"],
        "notes": "",
        "example_source_id": t.get("example_source_id", ""),
        "example_original_statement": t.get("example_original_statement", ""),
        "example_buggy_statement": t.get("example_buggy_statement", ""),
    })

    # The rules include manually resolved cases. If any NEEDS_REVIEW remain,
    # keep them visible for manual audit.
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote {args.output}")
    print(f"Mappings: {len(out)}")
    print(f"Needs review: {(out['needs_review'].astype(str).str.upper() == 'TRUE').sum()}")

if __name__ == "__main__":
    main()
