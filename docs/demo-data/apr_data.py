#!/usr/bin/env python3
import json
import argparse

def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Input .jsonl file (e.g., valid.jsonl)")
    ap.add_argument("--out", default="compliance_dataset.json", help="Output .json file")
    ap.add_argument("--num_samples", type=int, default=0,
                    help="Number of examples to keep from the start (0 = use all)")
    args = ap.parse_args()

    rows = load_jsonl(args.input)

    if args.num_samples > 0 and args.num_samples < len(rows):
        rows = rows[:args.num_samples]

    compliant = [ex.get("after", "").strip() for ex in rows if ex.get("after")]
    non_compliant = [ex.get("before", "").strip() for ex in rows if ex.get("before")]

    out_obj = {
        "compliant_responses": compliant,
        "non_compliant_responses": non_compliant,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out_obj, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(compliant)} compliant and {len(non_compliant)} non-compliant responses to {args.out}")

if __name__ == "__main__":
    main()

