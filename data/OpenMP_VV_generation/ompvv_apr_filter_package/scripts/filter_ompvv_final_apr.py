#!/usr/bin/env python3
"""
Filter an OMPVV taxonomy dataset into the cleaned APR benchmark subset.

What it does:
  1. Keeps only rows where the buggy mutation fails.
  2. Keeps only rows where the original code passes.
  3. Normalizes CHANGE_INDEX_EXPRESSION -> CHANGE_VARIABLE.
  4. Separates NEEDS_REVIEW rows for manual taxonomy review.
  5. Excludes ORACLE_MUTATION, CHANGE_SCAN, CHANGE_FORMAT_STRING, and NEEDS_REVIEW
     from the final APR-ready dataset.
  6. Writes sorted all/review/resolved/final outputs and summary/diagnostic CSVs.
  7. Optionally applies a manually reviewed mapping file by raw bug_type.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import pandas as pd


FINAL_APR_MUTATION_TYPES = [
    "REMOVE_CONSTRUCT",
    "REMOVE_CLAUSE",
    "REMOVE_SYNCHRONIZATION",
    "REMOVE_DEPENDENCY",
    "CHANGE_OPERATOR",
    "CHANGE_CONSTANT",
    "CHANGE_VARIABLE",
    "CHANGE_CLAUSE",
    "CHANGE_MAPPING",
    "CHANGE_RUNTIME_CALL",
    "CHANGE_LOOP_BOUND",
    "CHANGE_ASSIGNMENT",
    "REMOVE_STATEMENT",
    "CHANGE_CONFIGURATION",
    "CHANGE_CONDITION",
]

EXCLUDED_FROM_APR = {
    "ORACLE_MUTATION",
    "CHANGE_SCAN",
    "CHANGE_FORMAT_STRING",
    "NEEDS_REVIEW",
}

NORMALIZE_MUTATION_TYPE = {
    "CHANGE_INDEX_EXPRESSION": "CHANGE_VARIABLE",
}


TRUE_VALUES = {"true", "1", "yes", "y", "passed", "pass"}
FALSE_VALUES = {"false", "0", "no", "n", "failed", "fail", "compile_error", "error"}


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suffix in {".jsonl", ".ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"Unsupported input extension: {path.suffix}. Use CSV, JSON, or JSONL.")


def normalize_bool(value) -> bool | None:
    """Return True/False when parseable, otherwise None."""
    if pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    return None


def status_is_failed(value) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"failed", "fail", "compile_error", "error"}:
        return True
    if text in {"passed", "pass", "true", "1"}:
        return False
    return None


def status_is_passed(value) -> bool | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in {"passed", "pass", "true", "1"}:
        return True
    if text in {"failed", "fail", "compile_error", "error", "false", "0"}:
        return False
    return None


def choose_buggy_fails_mask(df: pd.DataFrame) -> pd.Series:
    """Prefer passed_most_recent_test when present; fall back to test_status."""
    if "passed_most_recent_test" in df.columns:
        parsed = df["passed_most_recent_test"].map(normalize_bool)
        mask = parsed.eq(False)
        if mask.notna().any():
            return mask.fillna(False)

    if "test_status" in df.columns:
        parsed = df["test_status"].map(status_is_failed)
        return parsed.eq(True).fillna(False)

    raise ValueError(
        "Cannot determine whether buggy code fails. Need passed_most_recent_test or test_status."
    )


def choose_original_passes_mask(df: pd.DataFrame) -> pd.Series:
    """Prefer passed_original_code when present; fall back to original_test_status."""
    if "passed_original_code" in df.columns:
        parsed = df["passed_original_code"].map(normalize_bool)
        mask = parsed.eq(True)
        if mask.notna().any():
            return mask.fillna(False)

    if "original_test_status" in df.columns:
        parsed = df["original_test_status"].map(status_is_passed)
        return parsed.eq(True).fillna(False)

    raise ValueError(
        "Cannot determine whether original code passes. Need passed_original_code or original_test_status."
    )


def require_columns(df: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def apply_review_updates(df: pd.DataFrame, review_updates_path: str | None) -> pd.DataFrame:
    """
    Apply a manually edited review CSV.

    Expected columns in review CSV:
      - bug_type
      - proposed_mutation_type

    Rows with blank proposed_mutation_type are ignored.
    """
    if not review_updates_path:
        return df

    updates = pd.read_csv(review_updates_path)
    require_columns(updates, ["bug_type", "proposed_mutation_type"])

    updates = updates.copy()
    updates["proposed_mutation_type"] = updates["proposed_mutation_type"].fillna("").astype(str).str.strip()
    updates = updates[updates["proposed_mutation_type"] != ""]

    if updates.empty:
        print("[INFO] Review update file has no non-empty proposed_mutation_type values; no changes applied.")
        return df

    allowed = set(FINAL_APR_MUTATION_TYPES) | EXCLUDED_FROM_APR | {"NEEDS_REVIEW"}
    bad = sorted(set(updates["proposed_mutation_type"]) - allowed)
    if bad:
        raise ValueError(
            "Review update file contains unknown proposed_mutation_type values: "
            f"{bad}\nAllowed: {sorted(allowed)}"
        )

    mapping = dict(zip(updates["bug_type"], updates["proposed_mutation_type"]))
    before_needs_review = (df["mutation_type"] == "NEEDS_REVIEW").sum()

    df = df.copy()
    df["mutation_type"] = df.apply(
        lambda row: mapping.get(row.get("bug_type"), row["mutation_type"]), axis=1
    )
    df["mutation_type"] = df["mutation_type"].replace(NORMALIZE_MUTATION_TYPE)

    after_needs_review = (df["mutation_type"] == "NEEDS_REVIEW").sum()
    print(f"[INFO] Applied review updates for {len(mapping)} raw bug_type values.")
    print(f"[INFO] NEEDS_REVIEW rows before/after updates: {before_needs_review}/{after_needs_review}")
    return df


def compact_text(series: pd.Series) -> str:
    for x in series:
        if pd.notna(x) and str(x).strip():
            return str(x)
    return ""


def make_unique_bug_type_review(review_df: pd.DataFrame) -> pd.DataFrame:
    cols = review_df.columns
    agg_spec = {"num_rows": ("bug_type", "size")}

    optional_first_cols = [
        ("example_source_id", "source_id"),
        ("example_category", "category"),
        ("example_bug_family", "bug_family"),
        ("example_original_statement", "original_statement"),
        ("example_buggy_statement", "buggy_statement"),
        ("example_targeted_oracle", "targeted_oracle"),
        ("example_expected_failure_mode", "expected_failure_mode"),
        ("example_file_name_updated", "file_name_updated"),
    ]

    for out_col, src_col in optional_first_cols:
        if src_col in cols:
            agg_spec[out_col] = (src_col, compact_text)

    if review_df.empty:
        base_cols = ["bug_type", "num_rows"] + [x[0] for x in optional_first_cols if x[1] in cols]
        out = pd.DataFrame(columns=base_cols)
    else:
        out = (
            review_df.groupby("bug_type", dropna=False)
            .agg(**agg_spec)
            .reset_index()
            .sort_values(["num_rows", "bug_type"], ascending=[False, True])
        )

    out["proposed_mutation_type"] = ""
    out["review_notes"] = ""
    return out


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"[WRITE] {path} ({len(df)} rows)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input taxonomy CSV/JSON/JSONL file.")
    parser.add_argument("--outdir", default="ompvv_final_apr_outputs", help="Output directory.")
    parser.add_argument(
        "--review-updates",
        default=None,
        help="Optional manually edited unique_bug_types_needs_review.csv with proposed_mutation_type filled in.",
    )
    parser.add_argument(
        "--prefix",
        default="ompvv",
        help="Prefix for output filenames.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df0 = read_table(args.input)
    print(f"[INFO] Loaded {len(df0)} rows from {args.input}")

    require_columns(df0, ["mutation_type", "bug_type"])

    df = df0.copy()
    df["mutation_type"] = df["mutation_type"].fillna("NEEDS_REVIEW").astype(str).str.strip()
    df["mutation_type"] = df["mutation_type"].replace(NORMALIZE_MUTATION_TYPE)

    df = apply_review_updates(df, args.review_updates)

    # Validation-status filtering.
    buggy_fails = choose_buggy_fails_mask(df)
    original_passes = choose_original_passes_mask(df)

    diagnostics_rows = []
    diagnostics_rows.append({"stage": "loaded", "num_rows": len(df)})
    diagnostics_rows.append({"stage": "buggy_fails", "num_rows": int(buggy_fails.sum())})
    diagnostics_rows.append({"stage": "buggy_passes_or_unknown_removed", "num_rows": int((~buggy_fails).sum())})
    diagnostics_rows.append({"stage": "original_passes", "num_rows": int(original_passes.sum())})
    diagnostics_rows.append({"stage": "original_fails_or_unknown_removed", "num_rows": int((~original_passes).sum())})

    clean = df[buggy_fails & original_passes].copy()
    diagnostics_rows.append({"stage": "clean_buggy_fail_original_pass", "num_rows": len(clean)})

    clean["taxonomy_status"] = clean["mutation_type"].apply(
        lambda x: "needs_review" if x == "NEEDS_REVIEW" else "resolved"
    )
    clean["include_in_apr"] = clean["mutation_type"].isin(FINAL_APR_MUTATION_TYPES)
    clean["excluded_from_apr_reason"] = ""
    clean.loc[clean["mutation_type"] == "NEEDS_REVIEW", "excluded_from_apr_reason"] = "needs_review"
    clean.loc[clean["mutation_type"] == "ORACLE_MUTATION", "excluded_from_apr_reason"] = "oracle_mutation"
    clean.loc[clean["mutation_type"].isin({"CHANGE_SCAN", "CHANGE_FORMAT_STRING"}), "excluded_from_apr_reason"] = "inactive_or_excluded_category"
    clean.loc[
        (~clean["mutation_type"].isin(FINAL_APR_MUTATION_TYPES))
        & (clean["excluded_from_apr_reason"] == ""),
        "excluded_from_apr_reason",
    ] = "outside_final_15"

    # Sort all rows with NEEDS_REVIEW at the top, then stable grouping.
    sort_cols = []
    clean["_needs_review_sort"] = clean["mutation_type"].eq("NEEDS_REVIEW")
    for c in ["_needs_review_sort", "mutation_type", "bug_type", "source_id", "file_name_updated"]:
        if c in clean.columns:
            sort_cols.append(c)
    ascending = [False] + [True] * (len(sort_cols) - 1)
    clean_sorted = clean.sort_values(sort_cols, ascending=ascending).drop(columns=["_needs_review_sort"])

    review_df = clean_sorted[clean_sorted["mutation_type"] == "NEEDS_REVIEW"].copy()
    resolved_df = clean_sorted[clean_sorted["mutation_type"] != "NEEDS_REVIEW"].copy()
    final_apr_df = clean_sorted[clean_sorted["include_in_apr"]].copy()
    excluded_df = clean_sorted[~clean_sorted["include_in_apr"]].copy()

    diagnostics_rows.extend([
        {"stage": "needs_review_rows", "num_rows": len(review_df)},
        {"stage": "resolved_rows", "num_rows": len(resolved_df)},
        {"stage": "final_apr_15_rows", "num_rows": len(final_apr_df)},
        {"stage": "excluded_from_apr_rows", "num_rows": len(excluded_df)},
    ])

    # Summaries.
    mutation_summary = (
        clean_sorted.groupby(["mutation_type", "taxonomy_status", "include_in_apr", "excluded_from_apr_reason"], dropna=False)
        .size()
        .reset_index(name="num_rows")
        .sort_values(["include_in_apr", "num_rows", "mutation_type"], ascending=[False, False, True])
    )

    final_apr_summary = (
        final_apr_df.groupby("mutation_type", dropna=False)
        .size()
        .reset_index(name="num_rows")
        .sort_values(["num_rows", "mutation_type"], ascending=[False, True])
    )

    unique_review = make_unique_bug_type_review(review_df)

    # Optional cross-tabs when columns exist.
    category_summary = pd.DataFrame()
    if "category" in clean_sorted.columns:
        category_summary = (
            final_apr_df.groupby(["category", "mutation_type"], dropna=False)
            .size()
            .reset_index(name="num_rows")
            .sort_values(["category", "num_rows"], ascending=[True, False])
        )

    family_summary = pd.DataFrame()
    if "bug_family" in clean_sorted.columns:
        family_summary = (
            final_apr_df.groupby(["bug_family", "mutation_type"], dropna=False)
            .size()
            .reset_index(name="num_rows")
            .sort_values(["bug_family", "num_rows"], ascending=[True, False])
        )

    # Duplicate filename diagnostics.
    duplicate_files = pd.DataFrame()
    if "file_name_updated" in clean_sorted.columns:
        duplicate_files = clean_sorted[
            clean_sorted["file_name_updated"].notna()
            & clean_sorted["file_name_updated"].duplicated(keep=False)
        ].sort_values("file_name_updated")

    unknown_types = sorted(set(clean_sorted["mutation_type"]) - set(FINAL_APR_MUTATION_TYPES) - EXCLUDED_FROM_APR)
    unknown_types_df = pd.DataFrame({"unknown_or_unexpected_mutation_type": unknown_types})

    # Writes.
    p = args.prefix
    write_csv(clean_sorted, outdir / f"{p}_clean_all_sorted.csv")
    write_csv(review_df, outdir / f"{p}_needs_review_only.csv")
    write_csv(unique_review, outdir / f"{p}_unique_bug_types_needs_review.csv")
    write_csv(resolved_df, outdir / f"{p}_resolved_non_review.csv")
    write_csv(final_apr_df, outdir / f"{p}_final_apr_15_categories.csv")
    write_csv(excluded_df, outdir / f"{p}_excluded_from_apr.csv")
    write_csv(mutation_summary, outdir / f"{p}_mutation_type_summary_clean_all.csv")
    write_csv(final_apr_summary, outdir / f"{p}_final_apr_15_summary.csv")
    write_csv(pd.DataFrame(diagnostics_rows), outdir / f"{p}_filter_diagnostics.csv")
    write_csv(unknown_types_df, outdir / f"{p}_unknown_mutation_types.csv")

    if not category_summary.empty:
        write_csv(category_summary, outdir / f"{p}_final_apr_by_category_and_mutation_type.csv")
    if not family_summary.empty:
        write_csv(family_summary, outdir / f"{p}_final_apr_by_bug_family_and_mutation_type.csv")
    if not duplicate_files.empty:
        write_csv(duplicate_files, outdir / f"{p}_duplicate_file_name_updated_rows.csv")

    # Console summary.
    print("\n========== FINAL SUMMARY ==========")
    print(f"Loaded rows:                         {len(df0)}")
    print(f"Clean rows, buggy fail + original pass: {len(clean_sorted)}")
    print(f"Needs-review rows:                  {len(review_df)}")
    print(f"Needs-review unique bug_type values: {review_df['bug_type'].nunique() if 'bug_type' in review_df.columns else 0}")
    print(f"Final APR rows, 15 categories:       {len(final_apr_df)}")
    print(f"Final APR active categories:         {final_apr_df['mutation_type'].nunique()}")
    if unknown_types:
        print(f"[WARN] Unexpected mutation types found: {unknown_types}")
    print("===================================\n")


if __name__ == "__main__":
    main()
