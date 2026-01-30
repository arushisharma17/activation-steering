#!/usr/bin/env python3
import argparse
import json

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer


def load_predictions(path):
    refs = []
    preds = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            refs.append(obj["reference"])
            preds.append(obj["prediction"])
    return refs, preds


def compute_bleu(references, predictions):
    smoothie = SmoothingFunction().method4
    scores = []
    for ref, pred in zip(references, predictions):
        ref_tokens = ref.split()
        pred_tokens = pred.split()
        if not pred_tokens:
            scores.append(0.0)
            continue
        score = sentence_bleu([ref_tokens], pred_tokens, smoothing_function=smoothie)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0


def compute_rouge_l(references, predictions):
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    f_scores = []
    for ref, pred in zip(references, predictions):
        scores = scorer.score(ref, pred)
        f_scores.append(scores["rougeL"].fmeasure)
    return sum(f_scores) / len(f_scores) if f_scores else 0.0


def try_codebleu(references, predictions, lang: str):
    """
    Try to compute CodeBLEU. If anything fails, return None and print a warning.
    """
    try:
        from codebleu import calc_codebleu
    except Exception as e:
        print(f"[WARN] Could not import codebleu (calc_codebleu): {e}")
        return None

    try:
        result = calc_codebleu(references, predictions, lang=lang)
        return result.get("codebleu", None)
    except Exception as e:
        print(f"[WARN] CodeBLEU computation failed: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_file", required=True)
    parser.add_argument("--lang", default="python",
                        help="Language for CodeBLEU (e.g., python, java, go).")
    args = parser.parse_args()

    refs, preds = load_predictions(args.pred_file)
    print(f"[INFO] Loaded {len(refs)} predictions from {args.pred_file}")

    bleu = compute_bleu(refs, preds)
    print(f"BLEU-4: {bleu:.4f}")

    rouge_l = compute_rouge_l(refs, preds)
    print(f"ROUGE-L (F1): {rouge_l:.4f}")

    codebleu_score = try_codebleu(refs, preds, lang=args.lang)
    if codebleu_score is not None:
        print(f"CodeBLEU: {codebleu_score:.4f}")
    else:
        print("CodeBLEU: N/A (see warnings above)")


if __name__ == "__main__":
    main()

