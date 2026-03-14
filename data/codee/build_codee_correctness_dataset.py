#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import requests

CORRECTNESS_IDS = [
    "PWR001",
    "PWR002",
    "PWR004",
    "PWR005",
    "PWR007",
    "PWR008",
    "PWR014",
    "PWR037",
    "PWR063",
    "PWR068",
    "PWR069",
    "PWR070",
    "PWR072",
    "PWR073",
    "PWR079",
    "PWR080",
    "PWR081",
    "PWR082",
    "PWR083",
    "PWR088",
    "PWR089",
    "PWD002",
    "PWD003",
    "PWD004",
    "PWD005",
    "PWD006",
    "PWD007",
    "PWD008",
    "PWD009",
    "PWD010",
    "PWD011",
    "RMK016",
]

RAW_BASE = "https://raw.githubusercontent.com/codee-com/open-catalog/main/Checks/{check_id}/README.md"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DROP_SINGLE_BLOCK_ROWS = True
SLEEP_SECONDS = 0.15

TITLE_RE = re.compile(r"#\s+([A-Z]{3}\d{3}:\s+.*?)(?=\s+###|\Z)", re.DOTALL)
LANG_MARKER_RE = re.compile(r"(####\s+(C\+\+|C|Fortran))", re.IGNORECASE)
CODE_BLOCK_RE = re.compile(r"```([A-Za-z0-9_+\-]*)\s+(.*?)```", re.DOTALL)


def download_readme(check_id: str) -> str | None:
    url = RAW_BASE.format(check_id=check_id)
    r = requests.get(url, headers=HEADERS, timeout=30)
    if r.status_code == 404:
        print(f"[WARN] Missing README for {check_id}")
        return None
    r.raise_for_status()
    return r.text


def extract_title(md: str, fallback: str) -> str:
    m = TITLE_RE.search(md)
    if m:
        return " ".join(m.group(1).split())
    return fallback


def get_code_example_region(md: str) -> str:
    start_marker = "### Code example"
    end_markers = ["### Related resources", "### References"]

    start = md.find(start_marker)
    if start == -1:
        return ""

    end = len(md)
    for marker in end_markers:
        pos = md.find(marker, start + len(start_marker))
        if pos != -1:
            end = min(end, pos)

    return md[start:end]


def normalize_language(lang: str) -> str | None:
    lang = lang.strip().lower()
    mapping = {
        "c": "C",
        "c++": "C++",
        "fortran": "Fortran",
    }
    return mapping.get(lang)


def extract_subsections(code_example_text: str) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = {}

    matches = list(LANG_MARKER_RE.finditer(code_example_text))
    print(f"    language markers found: {len(matches)}")

    if not matches:
        return grouped

    for i, match in enumerate(matches):
        lang_raw = match.group(2)
        language = normalize_language(lang_raw)
        if language is None:
            continue

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(code_example_text)
        subsection_text = code_example_text[start:end]

        blocks = []
        for code_match in CODE_BLOCK_RE.finditer(subsection_text):
            _fence_lang, code = code_match.groups()
            code = code.strip()
            if code:
                blocks.append(code)

        print(f"    {language}: {len(blocks)} code blocks")

        if blocks:
            grouped[language] = blocks

    return grouped


def build_rows(check_id: str, title: str, source_url: str, grouped: dict[str, list[str]]) -> list[dict]:
    rows = []
    for language, codes in grouped.items():
        if DROP_SINGLE_BLOCK_ROWS and len(codes) < 2:
            print(f"    skipping {language} because only {len(codes)} block")
            continue

        row = {
            "check_id": check_id,
            "title": title,
            "subsection_language": language,
            "source_url": source_url,
            "num_code_blocks_in_subsection": len(codes),
            "num_variants": max(0, len(codes) - 1),
            "base_code": codes[0],
        }

        for i, code in enumerate(codes[1:], start=1):
            row[f"corrected_code_{i}"] = code

        rows.append(row)

    return rows


def main() -> None:
    out_csv = Path("codee_correctness_pairs_by_subsection.csv")
    out_json = Path("codee_correctness_pairs_by_subsection.json")

    all_rows = []
    max_corrected_cols = 0

    for check_id in CORRECTNESS_IDS:
        try:
            print(f"[INFO] Processing {check_id}")
            md = download_readme(check_id)
            if not md:
                continue

            title = extract_title(md, check_id)
            code_example_region = get_code_example_region(md)
            print(f"    code example region length: {len(code_example_region)}")

            grouped = extract_subsections(code_example_region)
            rows = build_rows(check_id, title, RAW_BASE.format(check_id=check_id), grouped)
            print(f"    rows produced: {len(rows)}")

            all_rows.extend(rows)

            for row in rows:
                count = sum(1 for k in row if k.startswith("corrected_code_"))
                max_corrected_cols = max(max_corrected_cols, count)

            time.sleep(SLEEP_SECONDS)

        except Exception as e:
            print(f"[ERROR] {check_id}: {e}")

    fieldnames = [
        "check_id",
        "title",
        "subsection_language",
        "source_url",
        "num_code_blocks_in_subsection",
        "num_variants",
        "base_code",
    ] + [f"corrected_code_{i}" for i in range(1, max_corrected_cols + 1)]

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    out_json.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[DONE] Wrote {len(all_rows)} rows")
    print(f"[DONE] CSV: {out_csv}")
    print(f"[DONE] JSON: {out_json}")


if __name__ == "__main__":
    main()
