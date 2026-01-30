#!/usr/bin/env python
import argparse
import csv
import json
from pathlib import Path
from typing import List

# Optional metric deps
try:
    import sacrebleu  # BLEU
except ImportError:
    sacrebleu = None

try:
    from rouge_score import rouge_scorer  # ROUGE-L
except ImportError:
    rouge_scorer = None

try:
    from nltk.translate.meteor_score import meteor_score  # METEOR
except ImportError:
    meteor_score = None


def load_refs_hyps(path: Path) -> (List[str], List[str]):
    """Load reference and prediction text from a jsonl file."""
    refs, hyps = [], []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            # Expect exactly these keys
            if "reference" not in obj or "prediction" not in obj:
                continue

            refs.append(str(obj["reference"]))
            hyps.append(str(obj["prediction"]))
    if not refs:
        raise ValueError(f"No valid examples in {path} (missing reference/prediction?)")
    return refs, hyps


def compute_bleu(refs: List[str], hyps: List[str]):
    if sacrebleu is None:
        return "NA"
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    return float(bleu.score)


def compute_rougeL(refs: List[str], hyps: List[str]):
    if rouge_scorer is None:
        return "NA"
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = []
    for r, h in zip(refs, hyps):
        s = scorer.score(r, h)
        scores.append(s["rougeL"].fmeasure)
    if not scores:
        return "NA"
    return float(sum(scores) / len(scores) * 100.0)  # percentage


def compute_meteor(refs: List[str], hyps: List[str]):
    if meteor_score is None:
        return "NA"
    scores = []
    for r, h in zip(refs, hyps):
        scores.append(meteor_score([r.split()], h.split()))
    if not scores:
        return "NA"
    return float(sum(scores) / len(scores) * 100.0)  # percentage


def parse_tag_from_filename(fname: str):
    """
    For names like:
      codexglue_python_codellama7b_baseline.jsonl
      codexglue_python_qwen_coder7b_steered_l8_12_s2.5.jsonl

    Return (model_tag, condition_tag).
    """
    stem = fname
    if stem.endswith(".jsonl"):
        stem = stem[:-6]

    prefix = "codexglue_python_"
    if not stem.startswith(prefix):
        return stem, ""

    rest = stem[len(prefix):]

    for marker in ["_baseline", "_steered"]:
        if marker in rest:
            before, after = rest.split(marker, 1)
            model_tag = before
            cond_tag = marker.lstrip("_") + after
            return model_tag, cond_tag

    return rest, ""


def main():
    ap = argparse.ArgumentParser(
        description="Aggregate CodeXGLUE summarization results into a single CSV."
    )
    ap.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory with codexglue_python_*.jsonl files (default: results/)",
    )
    ap.add_argument(
        "--out_csv",
        type=str,
        default="summarization_summary_all_models.csv",
        help="Output CSV path (default: summarization_summary_all_models.csv)",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        raise SystemExit(f"[ERROR] results_dir does not exist: {results_dir}")

    jsonl_files = sorted(results_dir.glob("codexglue_python_*.jsonl"))
    if not jsonl_files:
        raise SystemExit(f"[ERROR] No codexglue_python_*.jsonl files found under {results_dir}")

    print(f"[INFO] Found {len(jsonl_files)} result files under {results_dir}")

    rows = []

    for path in jsonl_files:
        print(f"[INFO] Processing {path.name} ...")
        try:
            refs, hyps = load_refs_hyps(path)
        except Exception as e:
            print(f"[WARN] Skipping {path.name} due to error: {e}")
            continue

        bleu = compute_bleu(refs, hyps)
        rougeL = compute_rougeL(refs, hyps)
        meteor = compute_meteor(refs, hyps)

        model_tag, cond_tag = parse_tag_from_filename(path.name)

        row = {
            "file": path.name,
            "dataset": "codexglue_python",
            "model_tag": model_tag,
            "condition_tag": cond_tag,
            "num_examples": len(refs),
            "BLEU": bleu,
            "ROUGE_L": rougeL,
            "METEOR": meteor,
        }
        rows.append(row)

    if not rows:
        raise SystemExit("[ERROR] No rows to write; all files were skipped.")

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "file",
        "dataset",
        "model_tag",
        "condition_tag",
        "num_examples",
        "BLEU",
        "ROUGE_L",
        "METEOR",
    ]

    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"[INFO] Wrote summary for {len(rows)} files to {out_csv}")


if __name__ == "__main__":
    main()

