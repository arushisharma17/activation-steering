from __future__ import annotations
import re
from typing import Any, Dict

def norm(x: Any) -> str:
    if x is None:
        return ""
    return str(x).strip()

def infer_openmp_version(source_id: str) -> str:
    m = re.search(r"tests/(\d+\.\d+)/", norm(source_id))
    return m.group(1) if m else "unknown"

def infer_source_dir(source_id: str) -> str:
    parts = norm(source_id).split("/")
    if len(parts) >= 3:
        return "/".join(parts[:3])
    return norm(source_id)

def infer_file_feature_construct(source_id: str, category: str = "") -> str:
    """
    File-feature construct label.

    This is intentionally path/filename based. It represents the primary
    OpenMP feature tested by the OMPVV file, not necessarily the specific
    token changed by the mutation.
    """
    s = norm(source_id).lower()
    c = norm(category).lower()

    # Specific combined constructs first.
    if "/parallel_master_taskloop_simd/" in s:
        return "parallel_master_taskloop_simd"
    if "/parallel_master_taskloop/" in s:
        return "parallel_master_taskloop"
    if "/parallel_master/" in s:
        return "parallel_master"
    if "/parallel_for/" in s:
        return "parallel_for"

    # Filename-specific refinements.
    if "taskgraph" in s:
        return "taskgraph"
    if "taskloop_simd" in s:
        return "taskloop_simd"
    if "taskloop" in s:
        return "taskloop"
    if "taskwait" in s:
        return "taskwait"
    if "taskgroup" in s:
        return "taskgroup"
    if "/task/" in s:
        return "task"

    if "declare_mapper" in s:
        return "declare_mapper"
    if "declare_variant" in s:
        return "declare_variant"
    if "metadirective" in s:
        return "metadirective"
    if "dispatch" in s:
        return "dispatch"
    if "ordered" in s or "doacross" in s:
        return "ordered"

    if "atomic" in s:
        return "atomic"
    if "scan" in s:
        return "scan"
    if "reduction" in s and "/loop/" in s:
        return "loop"
    if "/loop/" in s or "loop_" in s or "unroll" in s:
        return "loop"
    if "tile" in s:
        return "tile"
    if "fuse" in s:
        return "fuse"

    if "target" in s or "map" in s or "mapped_ptr" in s or "requires" in s:
        return "target_mapping"
    if "teams" in s or "/teams/" in s:
        return "teams"
    if "simd" in s:
        return "simd"
    if "scope" in s:
        return "scope"
    if "assume" in s:
        return "assume"
    if "allocator" in s or "allocate" in s or "alloc" in s:
        return "allocator"
    if "flush" in s or "mem_order" in s or "memory_order" in s:
        return "memory_order"
    if "env_var" in s or "runtime_calls" in s or "affinity" in s or "program_control" in s:
        return "runtime_api"
    if "masked" in s:
        return "masked"
    if "error" in s:
        return "error"

    if c:
        return c
    parts = s.split("/")
    if len(parts) >= 3:
        return parts[2]
    return "unknown"

# Manually resolved labels from reviewed NEEDS_REVIEW list.
MANUAL_MUTATION_TYPE_OVERRIDES = {
    "BIND_PARALLEL_TO_THREAD": "CHANGE_CLAUSE",
    "BIND_TEAMS_TO_THREAD": "CHANGE_CLAUSE",
    "BIND_THREAD_TO_PARALLEL": "CHANGE_CLAUSE",
    "BIND_THREAD_TO_TEAMS": "CHANGE_CLAUSE",
    "BITAND_TO_BITOR": "CHANGE_OPERATOR",
    "BITOR_TO_BITAND": "CHANGE_OPERATOR",
    "BITXOR_TO_BITOR": "CHANGE_OPERATOR",
    "FINAL_ISFINAL_TO_NOT_ISFINAL": "CHANGE_CONDITION",
    "IF0_TO_IF1": "CHANGE_CONDITION",
    "LOGICAL_AND_TO_OR": "CHANGE_OPERATOR",
    "LOGICAL_OR_TO_AND": "CHANGE_OPERATOR",
    "MAX_TO_MIN": "CHANGE_OPERATOR",
    "MIN_TO_MAX": "CHANGE_OPERATOR",
    "PARALLEL_MASTER_TASKLOOP_SIMD_TO_PARALLEL_TASKLOOP_SIMD": "REMOVE_CLAUSE",
    "REL_OP_NEQ_TO_LT": "CHANGE_LOOP_BOUND",
    "SEVERITY_WARNING_TO_FATAL": "CHANGE_CLAUSE",
    "WRONG_ARRAY_UPDATE": "CHANGE_ASSIGNMENT",
    "WRONG_LOOP_BODY_ASSIGNMENT": "CHANGE_ASSIGNMENT",
    "WRONG_PARALLEL_VARIANT_SELECTOR": "CHANGE_CLAUSE",
    "WRONG_TARGET_VARIANT_SELECTOR": "CHANGE_CLAUSE",
    "WRONG_TIME_ASSIGNMENT": "CHANGE_ASSIGNMENT",
}

MANUAL_BUG_CONCEPT_OVERRIDES = {
    "BIND_PARALLEL_TO_THREAD": "bind",
    "BIND_TEAMS_TO_THREAD": "bind",
    "BIND_THREAD_TO_PARALLEL": "bind",
    "BIND_THREAD_TO_TEAMS": "bind",
    "BITAND_TO_BITOR": "reduction_operator",
    "BITOR_TO_BITAND": "reduction_operator",
    "BITXOR_TO_BITOR": "reduction_operator",
    "FINAL_ISFINAL_TO_NOT_ISFINAL": "task_final",
    "IF0_TO_IF1": "task_if",
    "LOGICAL_AND_TO_OR": "reduction_operator",
    "LOGICAL_OR_TO_AND": "reduction_operator",
    "MAX_TO_MIN": "reduction_operator",
    "MIN_TO_MAX": "reduction_operator",
    "PARALLEL_MASTER_TASKLOOP_SIMD_TO_PARALLEL_TASKLOOP_SIMD": "master",
    "REL_OP_NEQ_TO_LT": "loop_condition",
    "SEVERITY_WARNING_TO_FATAL": "severity",
    "WRONG_ARRAY_UPDATE": "loop_body",
    "WRONG_LOOP_BODY_ASSIGNMENT": "loop_body",
    "WRONG_PARALLEL_VARIANT_SELECTOR": "variant_selector",
    "WRONG_TARGET_VARIANT_SELECTOR": "variant_selector",
    "WRONG_TIME_ASSIGNMENT": "loop_body",
}

def classify_mutation_type(bug_type: str, original_statement: str = "", buggy_statement: str = "") -> str:
    b = norm(bug_type).upper()
    if b in MANUAL_MUTATION_TYPE_OVERRIDES:
        return MANUAL_MUTATION_TYPE_OVERRIDES[b]

    old = norm(original_statement)

    if any(tok in b for tok in ["WRONG_EXPECTED", "RETURN_VALUE", "ORACLE"]):
        return "ORACLE_MUTATION"

    if ("MAP_" in b or "MAPPER" in b or "ENTER_DATA" in b or "EXIT_DATA" in b
        or "MAPPED_PTR" in b or "TOFROM" in b or "FROM_TO" in b or "PRESENT" in b):
        return "CHANGE_MAPPING"

    if ("REMOVE_DEPEND" in b or "REMOVE_DOACROSS" in b or "REMOVE_IN_REDUCTION" in b
        or "REMOVE_TASK_REDUCTION" in b or ("IN_REDUCTION" in b and "REMOVE" in b)):
        return "REMOVE_DEPENDENCY"

    if ("REMOVE_ATOMIC" in b or "REMOVE_BARRIER" in b or "REMOVE_TASKWAIT" in b
        or "REMOVE_ORDERED" in b or "REMOVE_FLUSH" in b or "REMOVE_CANCEL" in b
        or "REMOVE_CRITICAL" in b or "ATOMIC_READ_TO_WRITE" in b or "REMOVE_MUTEX" in b):
        return "REMOVE_SYNCHRONIZATION"

    if ("REMOVE_TARGET" in b or "REMOVE_PARALLEL" in b or "REMOVE_TASKGRAPH" in b
        or "REMOVE_TASKLOOP" in b or "REMOVE_TASK" in b or "REMOVE_DISPATCH" in b
        or "REMOVE_SCOPE" in b or "REMOVE_SIMD" in b or "REMOVE_METADIRECTIVE" in b
        or "REMOVE_MASTER" in b or "REMOVE_MASKED" in b or "REMOVE_SINGLE" in b
        or "REMOVE_ERROR_DIRECTIVE" in b):
        return "REMOVE_CONSTRUCT"

    if b.startswith("REMOVE_") and any(tok in b for tok in [
        "PRIVATE", "FIRSTPRIVATE", "LASTPRIVATE", "DEFAULT", "REDUCTION", "NOWAIT",
        "TRANSPARENT", "SAFESYNC", "MESSAGE", "SEVERITY", "NOGROUP", "FILTER",
        "COLLAPSE", "AFFINITY", "ALIGNED", "ADJUST_ARGS", "UNROLL", "ASSUME"]):
        return "REMOVE_CLAUSE"

    if ("SKIP_LAST" in b or "SKIP_FIRST" in b or "LOOP_BOUND" in b
        or "WRONG_LOOP_RANGE" in b or "LOOPRANGE" in b):
        return "CHANGE_LOOP_BOUND"

    if "STRIDE" in b:
        return "CHANGE_CONSTANT"

    if "SCAN_" in b or "INCLUSIVE_TO_EXCLUSIVE" in b or "EXCLUSIVE_TO_INCLUSIVE" in b:
        return "CHANGE_SCAN"

    if ("_PLUS_TO_" in b or "_MINUS_TO_" in b or "_MULTIPLY" in b or
        "GREATER_TO_LESS" in b or "EQ_TO_NEQ" in b or "INVERT" in b or
        "WRONG_OPERATOR" in b or "COMPARE_" in b):
        return "CHANGE_OPERATOR"

    if ("IF_TO_" in b or "IF_CONDITION" in b or "CONDITION" in b or "PREDICATE" in b
        or "TRUE_TO_FALSE" in b or "FALSE_TO_TRUE" in b or "ALWAYS_FALSE" in b
        or "ALWAYS_TRUE" in b or "GRAPH_RESET" in b or "NOCONTEXT" in b or "NOVARIANTS" in b):
        return "CHANGE_CONDITION"

    if any(tok in b for tok in [
        "PRIVATE_TO_SHARED", "SHARED_TO_PRIVATE", "FIRSTPRIVATE_TO_PRIVATE",
        "FIRSTPRIVATE_TO_SHARED", "LASTPRIVATE_TO_PRIVATE", "LASTPRIVATE_TO_FIRSTPRIVATE",
        "DEFAULT_FIRSTPRIVATE_TO", "DEFAULT_PRIVATE_TO", "TO_PRIVATE", "TO_SHARED",
        "TO_FIRSTPRIVATE", "THREADSET_POOL_TO_TEAM", "MASTER_TO_PARALLEL", "MASKED_TO_MASTER"]):
        return "CHANGE_CLAUSE"

    if (b.startswith("WRONG_RUNTIME") or "RUNTIME" in b or "OMP_GET" in b or "OMP_SET" in b
        or "DISPLAY_ENV" in b or "ENV_" in b or "AFFINITY_FORMAT" in b or "SET_AFFINITY" in b
        or "GET_AFFINITY" in b):
        return "CHANGE_RUNTIME_CALL"

    if any(tok in b for tok in [
        "NUM_THREADS", "NUM_TEAMS", "THREAD_LIMIT", "NUM_TASKS", "GRAINSIZE", "TILE_SIZE",
        "ALIGNMENT", "ALLOC", "ALLOCATOR", "SIMDLEN", "FILTER_", "FORMAT_WIDTH",
        "TO_ONE", "TO_ZERO", "TO_FALSE", "TO_TRUE", "64_TO_32", "8_TO_7", "4_TO_2", "3_TO_4"]):
        return "CHANGE_CONFIGURATION"

    if any(tok in b for tok in [
        "WRONG_VARIABLE", "CHANGE_VARIABLE", "BUFFER_INDEX", "WRONG_INDEX",
        "WRONG_DEVICE", "TO_X_ONLY", "CONSTANT_GRAPH_ID"]):
        return "CHANGE_VARIABLE"

    if any(tok in b for tok in [
        "WRONG_CONSTANT", "WRONG_VALUE", "SET_Y_TO_WRONG_VALUE", "FORCE_ZERO", "FORCE_ONE",
        "TO_0", "TO_1", "TO_2", "TO_3", "TO_4"]):
        return "CHANGE_CONSTANT"

    if any(tok in b for tok in [
        "WRONG_ASSIGNMENT", "WRONG_INCREMENT", "WRONG_ACCUMULATION", "WRONG_UPDATE",
        "WRONG_RETURN", "WRONG_FIB", "WRONG_SCALE", "BASE_WRONG", "VARIANT_WRONG",
        "WRITE_WRONG"]):
        return "CHANGE_ASSIGNMENT"

    if b.startswith("REMOVE_"):
        if old.startswith("#pragma omp"):
            return "REMOVE_CONSTRUCT"
        return "REMOVE_STATEMENT"

    return "NEEDS_REVIEW"

def infer_bug_omp_concept(bug_type: str, original_statement: str = "", buggy_statement: str = "") -> str:
    b = norm(bug_type).upper()
    old = norm(original_statement).lower()
    joined = " ".join([b.lower(), old, norm(buggy_statement).lower()])

    if b in MANUAL_BUG_CONCEPT_OVERRIDES:
        return MANUAL_BUG_CONCEPT_OVERRIDES[b]
    if "map" in joined or "mapper" in joined or "present" in joined or "target enter data" in joined or "target exit data" in joined:
        return "mapping"
    if "depend" in joined or "doacross" in joined or "mutexinoutset" in joined:
        return "dependency"
    if "atomic" in joined:
        return "atomic"
    if "taskwait" in joined:
        return "taskwait"
    if "reduction" in joined or "in_reduction" in joined or "task_reduction" in joined or "inscan" in joined:
        return "reduction"
    if "private" in joined or "shared" in joined or "default" in joined:
        return "data_sharing"
    if "num_threads" in joined or "num_teams" in joined or "thread_limit" in joined:
        return "configuration"
    if "loop" in joined or "for (" in joined:
        return "loop"
    if "scan" in joined:
        return "scan"
    if "dispatch" in joined:
        return "dispatch"
    if "variant" in joined or "match(construct" in joined:
        return "variant_selector"
    if "severity" in joined or "message" in joined:
        return "message_or_severity"
    if "allocator" in joined or "alloc" in joined or "aligned" in joined:
        return "allocator"
    if "omp_" in joined:
        return "runtime_api"
    if "taskgraph" in joined or "graph_" in joined:
        return "taskgraph"
    return "general_code"

def classify_row(row: Dict[str, Any]) -> Dict[str, str]:
    bug_type = norm(row.get("bug_type", ""))
    source_id = norm(row.get("source_id", ""))
    category = norm(row.get("category", ""))
    original_statement = norm(row.get("original_statement", ""))
    buggy_statement = norm(row.get("buggy_statement", ""))

    mutation_type = classify_mutation_type(bug_type, original_statement, buggy_statement)
    omp_construct = infer_file_feature_construct(source_id, category)
    bug_omp_concept = infer_bug_omp_concept(bug_type, original_statement, buggy_statement)

    return {
        "bug_type": bug_type,
        "mutation_type": mutation_type,
        "omp_construct": omp_construct,
        "bug_omp_concept": bug_omp_concept,
        "source_dir": infer_source_dir(source_id),
        "openmp_version": infer_openmp_version(source_id),
        "needs_review": str(mutation_type == "NEEDS_REVIEW" or omp_construct == "unknown").upper(),
    }
