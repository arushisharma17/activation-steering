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

## Final Mutation Type Vocabulary

| Mutation Type | Meaning | Example |
|---------------|----------|---------|
| `REMOVE_CONSTRUCT` | Remove an OpenMP directive or full construct. | `#pragma omp target` → *(removed)* |
| `REMOVE_CLAUSE` | Remove a clause from an existing directive. | `reduction(+:sum)` → *(removed)* |
| `REMOVE_SYNCHRONIZATION` | Remove synchronization behavior such as `atomic`, `barrier`, `taskwait`, `ordered`, `flush`, or `cancel`. | `#pragma omp atomic` → *(removed)* |
| `REMOVE_DEPENDENCY` | Remove dependency or reduction-participation clauses. | `depend(in:x)` → *(removed)* |
| `CHANGE_OPERATOR` | Change an arithmetic, logical, comparison, bitwise, or reduction operator. | `reduction(max:result)` → `reduction(min:result)` |
| `CHANGE_CONSTANT` | Change a numeric constant or compile-time value. | `thread_limit(64)` → `thread_limit(32)` |
| `CHANGE_VARIABLE` | Change the variable, array, buffer, index, or reference being used. *(Not present in the current failing-only benchmark.)* | `depend(in:x)` → `depend(in:y)` |
| `CHANGE_CLAUSE` | Replace one OpenMP clause or clause argument with another. | `bind(parallel)` → `bind(thread)` |
| `CHANGE_MAPPING` | Change OpenMP target mapping behavior. | `map(tofrom:a)` → `map(to:a)` |
| `CHANGE_RUNTIME_CALL` | Change/remove OpenMP runtime API calls or arguments. | `omp_set_num_teams(8)` → `omp_set_num_teams(7)` |
| `CHANGE_LOOP_BOUND` | Change loop bounds/ranges so iterations are skipped or altered. | `i != N` → `i < N-1` |
| `CHANGE_ASSIGNMENT` | Change the computation or assignment in ordinary code. | `a[i] += b[i]` → `a[i] += b[i] + 1` |
| `REMOVE_STATEMENT` | Delete an ordinary non-pragma statement. | `sum += a[i];` → *(removed)* |
| `CHANGE_CONFIGURATION` | Change OpenMP execution configuration. | `num_threads(8)` → `num_threads(1)` |
| `CHANGE_CONDITION` | Change predicate expressions. | `if(0)` → `if(1)` |
| `CHANGE_SCAN` | Change scan-specific semantics. | `inclusive` → `exclusive` |
| `CHANGE_FORMAT_STRING` | Change format strings or affinity format tokens. *(Not present in the current failing-only benchmark.)* | `"OMP: %0.3f"` → `"OMP: %d"` |
| `ORACLE_MUTATION` | Change the test oracle/check itself. Usually excluded from APR training and evaluation. | `OMPVV_TEST_AND_SET(errors, x == 5)` → `OMPVV_TEST_AND_SET(errors, x == 4)` |

### Example Raw Bug Types

| Mutation Type | Example Raw Bug Types |
|---------------|----------------------|
| `REMOVE_CONSTRUCT` | `REMOVE_TARGET`, `REMOVE_TASK`, `REMOVE_TASKGRAPH`, `REMOVE_PARALLEL` |
| `REMOVE_CLAUSE` | `REMOVE_PRIVATE`, `REMOVE_FIRSTPRIVATE`, `REMOVE_REDUCTION`, `REMOVE_NOWAIT` |
| `REMOVE_SYNCHRONIZATION` | `REMOVE_ATOMIC`, `REMOVE_BARRIER`, `REMOVE_TASKWAIT`, `REMOVE_ORDERED` |
| `REMOVE_DEPENDENCY` | `REMOVE_DEPEND_IN`, `REMOVE_DEPEND_OUT`, `REMOVE_IN_REDUCTION`, `REMOVE_DOACROSS_SINK` |
| `CHANGE_OPERATOR` | `BITAND_TO_BITOR`, `MAX_TO_MIN`, `MIN_TO_MAX`, `LOGICAL_AND_TO_OR` |
| `CHANGE_CLAUSE` | `BIND_PARALLEL_TO_THREAD`, `PRIVATE_TO_SHARED`, `SEVERITY_WARNING_TO_FATAL` |
| `CHANGE_LOOP_BOUND` | `REL_OP_NEQ_TO_LT`, `SKIP_LAST_ITERATION` |
| `CHANGE_ASSIGNMENT` | `WRONG_ARRAY_UPDATE`, `WRONG_LOOP_BODY_ASSIGNMENT`, `WRONG_TIME_ASSIGNMENT` |
| `CHANGE_CONFIGURATION` | `NUM_THREADS_TO_ONE`, `NUM_TEAMS_TO_ONE`, `WRONG_TILE_SIZE` |
| `CHANGE_CONDITION` | `IF0_TO_IF1`, `FINAL_ISFINAL_TO_NOT_ISFINAL` |
| `CHANGE_MAPPING` | `MAP_TOFROM_TO_TO`, `MAP_FROM_TO_TO` |
| `ORACLE_MUTATION` | `WRONG_EXPECTED_VALUE`, `WRONG_RETURN_CHECK` |
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
