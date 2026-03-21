#!/usr/bin/env python3
"""
Central configuration for the HeCBench bug-introduction pipeline.

Bug categories are defined here as a simple dictionary so that they can be
easily added, removed, or reworded.  Everything in this file can also be
overridden via environment variables (see the bottom of the file).
"""

from __future__ import annotations

import os
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────
# Bug categories
# ──────────────────────────────────────────────────────────────────────
# Each entry maps a short ID → dict with three fields:
#
#   "keyword"   : short prompting keyword (used in the LLM prompt)
#   "ai_action" : concrete instruction for what the LLM should do
#
# ┌─────────────────────────┬──────────────────────────┬──────────────────────────────────────┐
# │ Bug Type                │ Prompting Keyword        │ AI Action                            │
# ├─────────────────────────┼──────────────────────────┼──────────────────────────────────────┤
# │ Data Race               │ Unprotected Shared Access│ Remove critical/atomic from a sum    │
# │ Race Condition          │ Incorrect Scoping        │ Move var from private() to shared()  │
# │ Deadlock                │ Orphaned Barrier         │ Barrier inside thread-id guard       │
# │ False Sharing           │ Cache Contention         │ Adjacent writes without padding      │
# │ Non-Determinism         │ Missing Ordered          │ Remove ordered from sequential loop  │
# │ ... (extended below)    │                          │                                      │
# └─────────────────────────┴──────────────────────────┴──────────────────────────────────────┘
#
# To add or remove categories, simply edit this dictionary.
# The pipeline scripts read it at runtime, so no other changes are needed.

BUG_CATEGORIES: dict[str, dict[str, str]] = {
    # ── Original 5 from the reference table ────────────────────────────
    "DATA_RACE": {
        "keyword": "Unprotected Shared Access",
        "ai_action": (
            "Remove a `#pragma omp critical` or `#pragma omp atomic` "
            "directive that protects an update to a shared accumulator "
            "(e.g. a sum, max, or counter).  The code must still compile "
            "but will produce nondeterministic results under parallel "
            "execution due to the unprotected concurrent write."
        ),
    },
    "RACE_CONDITION": {
        "keyword": "Incorrect Scoping",
        "ai_action": (
            "Move a loop-local variable from `private()` (or a local "
            "declaration inside the parallel region) to `shared()`, or "
            "add it to a `shared()` clause where it was previously "
            "correctly scoped as private.  The code must compile but "
            "give wrong results because multiple threads now share the "
            "same variable."
        ),
    },
    "DEADLOCK": {
        "keyword": "Orphaned Barrier",
        "ai_action": (
            "Place a `#pragma omp barrier` inside a conditional block "
            "that only some threads execute (e.g. inside an "
            "`if (omp_get_thread_num() == 0)` guard), so that not all "
            "threads in the team reach the barrier.  This will cause a "
            "deadlock or hang at runtime."
        ),
    },
    "FALSE_SHARING": {
        "keyword": "Cache Contention",
        "ai_action": (
            "Restructure data access so that different threads write to "
            "adjacent elements in a shared global array without any "
            "padding, causing cache-line contention (false sharing).  "
            "For example, replace per-thread local accumulators with "
            "direct writes to `result[omp_get_thread_num()]` in a "
            "tightly-packed array.  The code must compile and produce "
            "correct results, but will suffer severe performance "
            "degradation."
        ),
    },
    "NON_DETERMINISM": {
        "keyword": "Missing Ordered",
        "ai_action": (
            "Remove an `ordered` clause and/or `#pragma omp ordered` "
            "block from a loop that requires iterations to produce "
            "output in sequential order.  The code must compile but "
            "will produce output in a scrambled, run-dependent order."
        ),
    },
}

# ──────────────────────────────────────────────────────────────────────
# LLM configuration
# ──────────────────────────────────────────────────────────────────────
# If True, the script will spin up a local vLLM engine instead of making HTTP
# requests. This requires vllm to be installed (`pip install vllm`).
USE_LOCAL_VLLM: bool = os.environ.get("USE_LOCAL_VLLM", "True").lower() == "true"

# Point LLM_BASE_URL at your local vLLM / llama.cpp / Ollama endpoint if USE_LOCAL_VLLM is False it
LLM_BASE_URL: str = os.environ.get(
    "LLM_BASE_URL", "http://localhost:8000/v1"
)
LLM_API_KEY: str = os.environ.get("LLM_API_KEY", "EMPTY")
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")
LLM_TEMPERATURE: float = float(os.environ.get("LLM_TEMPERATURE", "0.7"))
LLM_MAX_TOKENS: int = int(os.environ.get("LLM_MAX_TOKENS", "4096"))

# ──────────────────────────────────────────────────────────────────────
# Source selection
# ──────────────────────────────────────────────────────────────────────
HECBENCH_REPO = "https://github.com/zjin-lcf/HeCBench.git"
BACKEND_SUFFIX = "-omp"  # filter for OpenMP benchmarks
SOURCE_EXTENSIONS: list[str] = [".cpp", ".c", ".h", ".hpp", ".cuh"]
MAX_SOURCE_LINES: int = int(os.environ.get("MAX_SOURCE_LINES", "500"))

# Default list of 100 OpenMP benchmarks to process.
# Edit this list freely; the fetch script will skip any name that does
# not have a matching `<name>-omp/` directory in the HeCBench repo.
DEFAULT_BENCHMARKS: list[str] = [
    "accuracy", "ace", "adam", "adamw", "adjacent",
    "adv", "aes", "affine", "aidw", "aligned-types",
    "all-pairs-distance", "amgmk", "ans", "aobench", "aop",
    "asmooth", "assert", "asta", "atan2", "backprop",
    "bfs", "bilateral", "binomial", "bm3d", "boxfilter",
    "bspline-vgh", "burger", "cbsfil", "ced", "cfd",
    "chemv", "chi2", "clenergy", "cobahh", "collision",
    "compute-score", "convolution1D", "convolution3D", "crc64", "crs",
    "d2q9_bgk", "damage", "dct8x8", "debayer", "diamond",
    "dp", "easyWave", "eigenvalue", "entropy", "extend2",
    "extrema", "fdtd3d", "fhd", "filter", "flame",
    "floydwarshall", "fluidSim", "fpc", "ga", "gaussian",
    "geodesic", "grep", "heartwall", "heat", "heat2d",
    "hellinger", "histogram", "hmm", "hotspot", "hotspot3D",
    "inversek2j", "is", "jacobi", "keogh", "kmeans",
    "knn", "laplace", "lavaMD", "leukocyte", "lid-driven-cavity",
    "logan", "lombscargle", "lr", "lud", "mandelbrot",
    "matrix-rotate", "maxpool3d", "md", "mdh", "meanshift",
    "medianfilter", "memcpy", "minimod", "minisweep", "miniWeather",
    "mriQ", "nbody", "nn", "nw", "page-rank",
]

# ──────────────────────────────────────────────────────────────────────
# Paths (relative to project root)
# ──────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent   # data/hecbench/
SOURCES_DIR = DATA_DIR / "sources"
MANIFEST_PATH = DATA_DIR / "sources" / "manifest.json"
RAW_BUGS_PATH = DATA_DIR / "raw_bugs.jsonl"
CONTRASTIVE_PAIRS_PATH = DATA_DIR / "contrastive_pairs_hecbench.json"
