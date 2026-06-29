# OMPVV APR Dataset Filter Package

This package filters the taxonomy CSV into the final cleaned APR benchmark subset discussed so far.

It does the following:

1. Keeps only rows where the buggy version fails.
2. Keeps only rows where the original version passes.
3. Normalizes `CHANGE_INDEX_EXPRESSION` to `CHANGE_VARIABLE`.
4. Groups all `NEEDS_REVIEW` rows together.
5. Writes a unique raw bug-type review sheet.
6. Excludes `ORACLE_MUTATION`, `CHANGE_SCAN`, `CHANGE_FORMAT_STRING`, and unresolved `NEEDS_REVIEW` rows from the final APR-ready 15-category dataset.

Final APR categories:

```text
REMOVE_CONSTRUCT
REMOVE_CLAUSE
REMOVE_SYNCHRONIZATION
REMOVE_DEPENDENCY
CHANGE_OPERATOR
CHANGE_CONSTANT
CHANGE_VARIABLE
CHANGE_CLAUSE
CHANGE_MAPPING
CHANGE_RUNTIME_CALL
CHANGE_LOOP_BOUND
CHANGE_ASSIGNMENT
REMOVE_STATEMENT
CHANGE_CONFIGURATION
CHANGE_CONDITION
```

## Colab usage

Upload these files to Colab:

- `ompvv_apr_filter_package.zip`
- your taxonomy CSV, for example `ompvv_bugs_failing_only_taxonomy (1).csv`

Then run:

```bash
!unzip -o ompvv_apr_filter_package.zip
%cd ompvv_apr_filter_package
!bash run_filter.sh "/content/ompvv_bugs_failing_only_taxonomy (1).csv" "/content/ompvv_final_apr_outputs"
```

## Main outputs

The output directory will contain:

```text
ompvv_clean_all_sorted.csv
ompvv_needs_review_only.csv
ompvv_unique_bug_types_needs_review.csv
ompvv_resolved_non_review.csv
ompvv_final_apr_15_categories.csv
ompvv_excluded_from_apr.csv
ompvv_mutation_type_summary_clean_all.csv
ompvv_final_apr_15_summary.csv
ompvv_filter_diagnostics.csv
ompvv_unknown_mutation_types.csv
```

The most important files are:

- `ompvv_unique_bug_types_needs_review.csv`: manually fill `proposed_mutation_type` for each unresolved raw `bug_type`.
- `ompvv_final_apr_15_categories.csv`: ready-to-use filtered APR benchmark subset.
- `ompvv_final_apr_15_summary.csv`: counts per final category.

## After manual review

Open `ompvv_unique_bug_types_needs_review.csv`, fill in `proposed_mutation_type`, save it, then rerun:

```bash
!bash run_filter.sh \
  "/content/ompvv_bugs_failing_only_taxonomy (1).csv" \
  "/content/ompvv_final_apr_outputs_reviewed" \
  "/content/ompvv_final_apr_outputs/ompvv_unique_bug_types_needs_review.csv"
```

Only non-empty `proposed_mutation_type` values are applied.

Allowed values for `proposed_mutation_type` are the 15 final APR categories plus:

```text
NEEDS_REVIEW
ORACLE_MUTATION
CHANGE_SCAN
CHANGE_FORMAT_STRING
```

Rows assigned to `ORACLE_MUTATION`, `CHANGE_SCAN`, or `CHANGE_FORMAT_STRING` are kept in the sorted/diagnostic files but excluded from `ompvv_final_apr_15_categories.csv`.
