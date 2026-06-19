# OMPVV Failing-Bug Taxonomy Tools — Final File-Feature Version

This directory contains the revised taxonomy workflow for the OpenMP_VV synthetic bug-fixing dataset.

The original dataset is **not modified**. The pipeline:

1. filters to validated test-failing bugs only,
2. separates passing mutations into a rejected file,
3. builds a raw `bug_type` mapping,
4. adds standardized mutation labels,
5. derives file-feature-based OpenMP construct labels,
6. computes statistics.

## Main design decision

The final schema separates three ideas that were previously mixed:

```text
bug_type          = fine-grained raw synthetic mutation name
mutation_type     = standardized mutation operation
omp_construct     = primary OpenMP feature tested by the OMPVV file
bug_omp_concept   = specific OpenMP concept affected by the introduced mutation
```

This means:

```text
source_id: tests/5.0/loop/test_loop_bind.c
bug_type:  BIND_PARALLEL_TO_THREAD

omp_construct   = loop
bug_omp_concept = bind
mutation_type   = CHANGE_CLAUSE
```

`omp_construct` is intentionally file/path based. It answers:

> What OpenMP feature is this OMPVV test file primarily about?

`bug_omp_concept` is mutation based. It answers:

> What OpenMP concept did the introduced bug directly alter?

## Failing-only benchmark

The original uploaded dataset has 372 examples. This workflow keeps only:

```text
passed_most_recent_test == false
```

Those are the validated test-failing bugs suitable for APR / bug-fixing evaluation.

Passing mutations are written separately as rejected candidates.

## Final mutation type vocabulary

| mutation_type | Meaning |
|---|---|
| `REMOVE_CONSTRUCT` | Remove an OpenMP directive or full construct. |
| `REMOVE_CLAUSE` | Remove a clause from an existing directive. |
| `REMOVE_SYNCHRONIZATION` | Remove synchronization behavior such as `atomic`, `barrier`, `taskwait`, `ordered`, `flush`, or `cancel`. |
| `REMOVE_DEPENDENCY` | Remove dependency or reduction-participation clauses such as `depend`, `in_reduction`, `task_reduction`, or doacross source/sink. |
| `CHANGE_OPERATOR` | Change an arithmetic, logical, comparison, bitwise, or reduction operator. |
| `CHANGE_CONSTANT` | Change a numeric constant or compile-time value. |
| `CHANGE_VARIABLE` | Change the variable, array, buffer, index, or reference being used. This category is allowed but currently not present in the failing-only benchmark. |
| `CHANGE_CLAUSE` | Replace one OpenMP clause or clause argument with another. |
| `CHANGE_MAPPING` | Change OpenMP target mapping behavior such as `tofrom -> to`, `from -> to`, `from -> delete`, or target data mapping. |
| `CHANGE_RUNTIME_CALL` | Change/remove OpenMP runtime API calls or their arguments. |
| `CHANGE_LOOP_BOUND` | Change loop bounds/ranges so iterations are skipped or altered. |
| `CHANGE_ASSIGNMENT` | Change the computation or assignment in ordinary code. |
| `REMOVE_STATEMENT` | Delete an ordinary non-pragma statement. |
| `CHANGE_CONFIGURATION` | Change OpenMP execution configuration such as `num_threads`, `num_teams`, `thread_limit`, `grainsize`, `tile sizes`, or allocator alignment. |
| `CHANGE_CONDITION` | Change predicate expressions such as `if`, `final`, `nocontext`, `novariants`, `graph_reset`, or assume predicates. |
| `CHANGE_SCAN` | Change scan-specific semantics such as `inclusive -> exclusive`. |
| `CHANGE_FORMAT_STRING` | Change format strings or affinity format tokens. This category is allowed but currently not present in the failing-only benchmark. |
| `ORACLE_MUTATION` | Change the test oracle/check itself. Usually exclude from APR training or report separately. |

There should be no `NEEDS_REVIEW` labels in the final mapping. If any appear, the mapping is incomplete.

## File-feature `omp_construct`

`omp_construct` is inferred from the OMPVV path and filename.

Examples:

| source_id | omp_construct |
|---|---|
| `tests/5.0/loop/test_loop_bind.c` | `loop` |
| `tests/5.0/loop/test_loop_reduction_max.c` | `loop` |
| `tests/5.1/declare_variant/test_begin_end_declare_variant.c` | `declare_variant` |
| `tests/5.0/parallel_for/test_parallel_for_notequals.c` | `parallel_for` |
| `tests/5.2/ordered/test_ordered_doacross.c` | `ordered` |
| `tests/5.1/tile/test_tile.c` | `tile` |
| `tests/6.0/taskgraph/test_taskgraph_id.c` | `taskgraph` |

## Mutation-detail `bug_omp_concept`

`bug_omp_concept` is inferred from `bug_type`, `original_statement`, and `buggy_statement`.

Examples:

| bug_type | bug_omp_concept |
|---|---|
| `BIND_PARALLEL_TO_THREAD` | `bind` |
| `BITAND_TO_BITOR` | `reduction_operator` |
| `FINAL_ISFINAL_TO_NOT_ISFINAL` | `task_final` |
| `WRONG_PARALLEL_VARIANT_SELECTOR` | `variant_selector` |
| `REL_OP_NEQ_TO_LT` | `loop_condition` |
| `PARALLEL_MASTER_TASKLOOP_SIMD_TO_PARALLEL_TASKLOOP_SIMD` | `master` |

## Workflow

### 1. Filter validated bugs

```bash
python scripts/00_filter_validated_bugs.py \
  --input data/ompvv_bugs_june.json \
  --output-valid outputs/ompvv_bugs_failing_only.json \
  --output-rejected outputs/ompvv_bugs_passing_rejected.json \
  --summary outputs/filter_summary.csv
```

### 2. Extract unique raw bug types

```bash
python scripts/01_extract_bug_types.py \
  --input outputs/ompvv_bugs_failing_only.json \
  --output outputs/bug_type_template_failing_only.csv
```

### 3. Build reviewed mapping

```bash
python scripts/02_build_mapping.py \
  --template outputs/bug_type_template_failing_only.csv \
  --output outputs/bug_type_mapping_final_failing_only.csv
```

This step uses rule-based classification plus the manually resolved `NEEDS_REVIEW` cases.

### 4. Apply taxonomy labels

```bash
python scripts/03_apply_taxonomy.py \
  --input outputs/ompvv_bugs_failing_only.json \
  --mapping outputs/bug_type_mapping_final_failing_only.csv \
  --output-json outputs/ompvv_bugs_failing_only_taxonomy.json \
  --output-csv outputs/ompvv_bugs_failing_only_taxonomy.csv
```

### 5. Compute statistics

```bash
python scripts/04_compute_taxonomy_stats.py \
  --input outputs/ompvv_bugs_failing_only_taxonomy.csv \
  --outdir outputs/stats_failing_only
```

### 6. Run all steps

```bash
bash run_taxonomy_pipeline_failing_only.sh data/ompvv_bugs_june.json outputs
```

## Key outputs

```text
outputs/ompvv_bugs_failing_only_taxonomy.csv
outputs/ompvv_bugs_failing_only_taxonomy.json
outputs/bug_type_mapping_final_failing_only.csv
outputs/ompvv_bugs_passing_rejected.json
outputs/filter_summary.csv
outputs/stats_failing_only/*.csv
```

## Recommended APR subset

For a strict repair benchmark, use the failing-only dataset and optionally exclude oracle mutations:

```python
df = df[df["mutation_type"] != "ORACLE_MUTATION"]
```

The oracle-mutation rows are useful for audit, but they are not ideal APR training/evaluation examples because they modify the test check rather than the program behavior.
