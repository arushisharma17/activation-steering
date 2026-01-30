# gen.py : HumanEval generation with optional activation steering and dataset/condition tags
import argparse
import os
from human_eval.data import write_jsonl, read_problems

from utils import (
    generate,
    load_model_and_tokenizer,
    maybe_apply_steering,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        required=True,
        help="Hugging Face model path or identifier.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help=(
            "Output JSONL filename (will be placed under results/). "
            "If not provided, auto-named using model, n, dataset-tag, condition, and steering flag."
        ),
    )
    ap.add_argument(
        "--n",
        type=int,
        default=1,
        help="Samples per HumanEval task.",
    )
    ap.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate.",
    )
    ap.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature.",
    )
    ap.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Nucleus sampling top-p.",
    )

    # Dataset / condition tags for nicer naming
    ap.add_argument(
        "--dataset-tag",
        default=None,
        help=(
            "Optional dataset tag for naming (e.g., 'tssb', 'manysstubs'). "
            "This is only used for the output filename."
        ),
    )
    ap.add_argument(
        "--condition",
        default=None,
        help=(
            "Optional condition name for naming (e.g., 'baseline', 'refusal-tssb'). "
            "This is only used for the output filename."
        ),
    )

    # Steering options
    ap.add_argument(
        "--steer",
        action="store_true",
        help="Enable activation steering using a SteeringVector.",
    )
    ap.add_argument(
        "--vector_path",
        default="refusal_behavior_vector",
        help="Path to SteeringVector file (e.g. 'refusal_behavior_vector').",
    )
    ap.add_argument(
        "--strength",
        type=float,
        default=2.0,
        help="Steering vector strength.",
    )
    ap.add_argument(
        "--layers",
        default="last:4",
        help="Layer spec: 'all', 'last:k', or comma-separated indices.",
    )

    args = ap.parse_args()

    # -----------------------------
    # Output naming + results dir
    # -----------------------------
    if args.out is None:
        slug = args.model.rstrip("/").split("/")[-1]

        parts = ["humaneval", f"n{args.n}"]

        # dataset tag (e.g., tssb, manysstubs)
        if args.dataset_tag:
            parts.append(args.dataset_tag)

        # steering / condition
        if args.steer:
            parts.append("steered")
            if args.condition:
                parts.append(args.condition)
        else:
            # baseline or other condition without steering
            if args.condition:
                parts.append(args.condition)

        filename = f"{slug}_{'_'.join(parts)}.jsonl"
    else:
        filename = args.out

    results_dir = "results_100"
    os.makedirs(results_dir, exist_ok=True)
    args.out = os.path.join(results_dir, filename)

    print(f"[INFO] Output file: {args.out}")
    print(f"[INFO] Loading base model: {args.model}")
    if args.dataset_tag:
        print(f"[INFO] Dataset tag: {args.dataset_tag}")
    if args.condition:
        print(f"[INFO] Condition: {args.condition}")
    print(f"[INFO] Steering: {'ON' if args.steer else 'OFF'}")

    # -----------------------------
    # Load model + tokenizer
    # -----------------------------
    base_model, tok = load_model_and_tokenizer(args.model)

    # -----------------------------
    # Optional steering
    # -----------------------------
    model_for_gen, used_layers = maybe_apply_steering(
        base_model=base_model,
        tokenizer=tok,
        steer=args.steer,
        vector_path=args.vector_path,
        strength=args.strength,
        layers=args.layers,
    )
    if used_layers is not None:
        print(f"[INFO] Steering active on layers: {used_layers}")
    else:
        print("[INFO] No steering (baseline).")

    # -----------------------------
    # HumanEval generation
    # -----------------------------
    problems = read_problems()
    samples = []

    print(f"[INFO] Generating {args.n} sample(s) per task for {len(problems)} tasks...")
    for i, (tid, prob) in enumerate(problems.items(), start=1):
        if i % 20 == 0:
            print(f"[INFO] {i}/{len(problems)} tasks done")
        for _ in range(args.n):
            comp = generate(
                model_for_gen,
                tok,
                prob["prompt"],
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            samples.append({"task_id": tid, "completion": comp})

    write_jsonl(args.out, samples)
    print(f"[INFO] Done. Saved {len(samples)} samples to {args.out}")


if __name__ == "__main__":
    main()

