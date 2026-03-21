#!/usr/bin/env python3
"""
Fetch HeCBench OpenMP benchmark source files.

Does a sparse git checkout of the HeCBench repo, copies qualifying source
files to a flat working directory, and writes a manifest JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from config import (
    BACKEND_SUFFIX,
    DEFAULT_BENCHMARKS,
    HECBENCH_REPO,
    MANIFEST_PATH,
    MAX_SOURCE_LINES,
    SOURCE_EXTENSIONS,
    SOURCES_DIR,
)


def clone_sparse(repo_url: str, dest: Path) -> None:
    """Sparse-checkout only the src/ directory of HeCBench."""
    if dest.exists():
        print(f"[INFO] Repo already cloned at {dest}, skipping clone.")
        return

    print(f"[INFO] Sparse-cloning {repo_url} into {dest} ...")
    env = os.environ.copy()
    env["GIT_LFS_SKIP_SMUDGE"] = "1"  # Skip downloading large dataset blobs

    subprocess.run(
        [
            "git", "clone",
            "--filter=blob:none",
            "--sparse",
            "--depth=1",
            repo_url,
            str(dest),
        ],
        check=True,
        env=env,
    )
    subprocess.run(
        ["git", "sparse-checkout", "set", "src"],
        cwd=str(dest),
        check=True,
        env=env,
    )
    print("[INFO] Sparse checkout complete.")


def collect_sources(
    repo_dir: Path,
    benchmarks: list[str],
    out_dir: Path,
    max_lines: int,
    extensions: list[str],
) -> list[dict]:
    """Copy benchmark directories and return manifest of valid OpenMP files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    src_root = repo_dir / "src"
    manifest: list[dict] = []
    skipped_missing = 0
    skipped_large = 0
    skipped_no_omp = 0

    for bench_name in benchmarks:
        bench_dir = src_root / f"{bench_name}{BACKEND_SUFFIX}"
        if not bench_dir.is_dir():
            print(f"[WARN] No directory for {bench_name}: {bench_dir}")
            skipped_missing += 1
            continue

        # Copy only source files to preserve structure without grabbing huge data files
        dest_bench_dir = out_dir / bench_name
        if dest_bench_dir.exists():
            shutil.rmtree(dest_bench_dir)
        dest_bench_dir.mkdir(parents=True, exist_ok=True)

        best_manifest_entry = None
        best_line_count = 0

        for fpath in bench_dir.rglob("*"):
            if not fpath.is_file():
                continue
            if fpath.suffix.lower() not in extensions:
                continue

            # Create destination subdir if needed
            rel_path = fpath.relative_to(bench_dir)
            dest_fpath = dest_bench_dir / rel_path
            dest_fpath.parent.mkdir(parents=True, exist_ok=True)

            try:
                shutil.copy2(fpath, dest_fpath)
            except Exception as exc:
                print(f"[WARN] Cannot copy {fpath}: {exc}")
                continue

            try:
                content = dest_fpath.read_text(encoding="utf-8", errors="replace")
                lines = content.splitlines()
            except Exception as exc:
                print(f"[WARN] Cannot read {dest_fpath}: {exc}")
                continue

            if len(lines) > max_lines:
                skipped_large += 1
                continue

            # Must actually contain OpenMP semantics to be useful for OpenMP bugs
            if "omp" not in content.lower():
                skipped_no_omp += 1
                continue

            line_count = len(lines)
            if line_count > best_line_count:
                best_line_count = line_count
                best_manifest_entry = {
                    "benchmark": bench_name,
                    "source_file": str(dest_fpath.relative_to(out_dir)),
                    "line_count": line_count,
                    "language": "C++" if fpath.suffix in (".cpp", ".h", ".hpp", ".cuh") else "C",
                }

        if best_manifest_entry:
            manifest.append(best_manifest_entry)

    print(f"[INFO] Collected {len(manifest)} valid OpenMP files from {len(benchmarks)} benchmarks")
    print(f"[INFO] Skipped (but still copied): {skipped_large} too-large files, {skipped_no_omp} non-OpenMP files")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Fetch HeCBench sources")
    ap.add_argument(
        "--benchmarks",
        type=lambda s: s.split(","),
        default=None,
        help="Comma-separated benchmark names (default: use config.DEFAULT_BENCHMARKS)",
    )
    ap.add_argument("--out", type=Path, default=SOURCES_DIR)
    ap.add_argument("--repo-dir", type=Path, default=Path("/tmp/hecbench_repo"))
    ap.add_argument("--max-lines", type=int, default=MAX_SOURCE_LINES)
    args = ap.parse_args()

    benchmarks = args.benchmarks or DEFAULT_BENCHMARKS

    # Step 1 — clone / update
    clone_sparse(HECBENCH_REPO, args.repo_dir)

    # Step 2 — collect sources
    manifest = collect_sources(
        repo_dir=args.repo_dir,
        benchmarks=benchmarks,
        out_dir=args.out,
        max_lines=args.max_lines,
        extensions=SOURCE_EXTENSIONS,
    )

    # Step 3 — write manifest
    manifest_path = args.out / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[DONE] Manifest written to {manifest_path}  ({len(manifest)} entries)")


if __name__ == "__main__":
    main()
