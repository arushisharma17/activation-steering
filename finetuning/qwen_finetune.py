#!/usr/bin/env python3
"""
qwen_finetune.py

QLoRA fine-tuning for Qwen/Qwen2.5-* models on a local JSONL bug-fix dataset.

Each JSONL line must contain at least:
  {"before": "...", "after": "..."}
Optionally:
  {"instruction": "..."}  # if not present, a default instruction is used.

Key behaviors:
- SFT-style label masking: loss computed ONLY on the "after" tokens (+ EOS), prompt tokens get -100.
- Truncation preserves the entire response whenever possible (truncates prompt first).
- Uses dynamic padding in the collator to avoid wasting compute when responses are short.
- Saves LoRA adapters + tokenizer + metrics + loss curve plot.
"""

import os
import json
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple

import torch
import matplotlib.pyplot as plt

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
    set_seed,
)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


# ----------------------------
# Utilities
# ----------------------------
def load_jsonl_records(path: str, default_instruction: str) -> Dataset:
    """
    Load JSONL -> HF Dataset.
    Produces fields: instruction, context, response
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except Exception as e:
                raise ValueError(f"Failed to parse JSON on line {line_no} of {path}: {e}")

            before = (ex.get("before") or "").strip()
            after = (ex.get("after") or "").strip()
            if not before or not after:
                continue

            instr = (ex.get("instruction") or default_instruction).strip()

            records.append(
                {
                    "instruction": instr,
                    "context": before,
                    "response": after,
                }
            )

    if len(records) == 0:
        raise ValueError(f"No usable records found in {path}. Expect JSONL with 'before' and 'after'.")
    print(f"[INFO] Loaded {len(records)} examples from {path}")
    return Dataset.from_list(records)


def build_prompt(instruction: str, context: str) -> str:
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Context:\n"
        f"{context}\n\n"
        "### Response:\n"
    )


def truncate_preserve_response(
    prompt_ids: List[int],
    resp_ids: List[int],
    eos_id: int,
    max_length: int,
) -> Tuple[List[int], List[int]]:
    """
    Build input_ids = prompt + response + eos
    labels   = -100 for prompt, response token ids for response, eos_id for eos

    Truncation strategy:
    - Prefer preserving response (supervised part).
    - If too long, truncate prompt from the left.
    - If response alone is too long, truncate response from the right.
    """
    max_wo_eos = max_length - 1
    if max_wo_eos <= 0:
        raise ValueError("max_length must be >= 2.")

    if len(resp_ids) > max_wo_eos:
        resp_ids = resp_ids[:max_wo_eos]
        prompt_ids = []
    else:
        remaining_for_prompt = max_wo_eos - len(resp_ids)
        if len(prompt_ids) > remaining_for_prompt:
            prompt_ids = prompt_ids[-remaining_for_prompt:]

    input_ids = prompt_ids + resp_ids + [eos_id]
    labels = ([-100] * len(prompt_ids)) + resp_ids + [eos_id]
    return input_ids, labels


def tokenize_sft_batch(
    examples: Dict[str, List[str]],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize prompt+response with label masking.
    NOTE: no padding here; we pad dynamically in the collator.
    """
    input_ids_batch = []
    labels_batch = []

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id.")

    for instr, ctx, resp in zip(examples["instruction"], examples["context"], examples["response"]):
        prompt = build_prompt(instr, ctx)
        prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
        resp_ids = tokenizer(resp, add_special_tokens=False).input_ids

        input_ids, labels = truncate_preserve_response(
            prompt_ids=prompt_ids,
            resp_ids=resp_ids,
            eos_id=eos_id,
            max_length=max_length,
        )
        input_ids_batch.append(input_ids)
        labels_batch.append(labels)

    return {"input_ids": input_ids_batch, "labels": labels_batch}


@dataclass
class DynamicSFTCollator:
    """
    Dynamic left-padding collator (matches tokenizer.padding_side='left').

    We pad input_ids with pad_token_id.
    We pad labels with -100.
    We create attention_mask accordingly.
    """
    tokenizer: AutoTokenizer

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            raise ValueError("Tokenizer has no pad_token_id; set tokenizer.pad_token.")

        max_len = max(len(f["input_ids"]) for f in features)

        input_ids_out = []
        labels_out = []
        attention_out = []

        for f in features:
            ids = f["input_ids"]
            labs = f["labels"]
            pad_len = max_len - len(ids)

            if self.tokenizer.padding_side == "left":
                ids_p = ([pad_id] * pad_len) + ids
                labs_p = ([-100] * pad_len) + labs
                attn = ([0] * pad_len) + ([1] * len(ids))
            else:
                ids_p = ids + ([pad_id] * pad_len)
                labs_p = labs + ([-100] * pad_len)
                attn = ([1] * len(ids)) + ([0] * pad_len)

            input_ids_out.append(ids_p)
            labels_out.append(labs_p)
            attention_out.append(attn)

        return {
            "input_ids": torch.tensor(input_ids_out, dtype=torch.long),
            "labels": torch.tensor(labels_out, dtype=torch.long),
            "attention_mask": torch.tensor(attention_out, dtype=torch.long),
        }


def plot_losses(trainer: Trainer, out_path: str) -> None:
    train_pts = []
    eval_pts = []
    for log in trainer.state.log_history:
        if "loss" in log and "learning_rate" in log:
            train_pts.append((log["step"], log["loss"]))
        if "eval_loss" in log:
            eval_pts.append((log["step"], log["eval_loss"]))

    if not train_pts and not eval_pts:
        print("[WARN] No losses found in log_history; skipping plot.")
        return

    plt.figure(figsize=(12, 6))
    if train_pts:
        xs, ys = zip(*train_pts)
        plt.plot(xs, ys, marker=".", label="Training Loss")
    if eval_pts:
        xs, ys = zip(*eval_pts)
        plt.plot(xs, ys, marker="o", linestyle="--", label="Eval Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title("Training and Evaluation Loss Over Steps")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path)
    print(f"[INFO] Loss plot saved to {out_path}")


def sanity_check_tokenization(ds: Dataset, tokenizer: AutoTokenizer, n: int = 1) -> None:
    for i in range(min(n, len(ds))):
        ex = ds[i]
        labels = ex["labels"]
        input_ids = ex["input_ids"]

        supervised = sum(1 for t in labels if t != -100)
        total = len(labels)
        frac = supervised / max(total, 1)

        first_sup = next((j for j, t in enumerate(labels) if t != -100), None)

        print(f"[SANITY] Example {i}: supervised_tokens={supervised}/{total} ({frac:.3f})")
        if first_sup is not None:
            lo = max(0, first_sup - 80)
            hi = min(total, first_sup + 120)
            snippet_txt = tokenizer.decode(input_ids[lo:hi], skip_special_tokens=False)
            print("[SANITY] Decode around supervision boundary:\n")
            print(snippet_txt)
            print("\n" + "-" * 80 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-Coder-7B-Instruct")
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)

    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--eval_ratio", type=float, default=0.10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--default_instruction", type=str, default="Fix the following buggy code snippet.")

    # training
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_ratio", type=float, default=0.10)
    p.add_argument("--train_bs", type=int, default=2)
    p.add_argument("--eval_bs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=16)

    p.add_argument("--eval_steps", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=400)  # default is multiple of eval_steps
    p.add_argument("--log_steps", type=int, default=50)
    p.add_argument("--early_stopping_patience", type=int, default=3)

    # precision / memory
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")

    # LoRA
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument(
        "--lora_targets",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names.",
    )

    p.add_argument("--sanity_n", type=int, default=1)

    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    set_seed(args.seed)

    device_map = "auto" if torch.cuda.is_available() else "cpu"
    if device_map == "cpu":
        print("[WARN] CUDA not available; CPU training will be extremely slow.")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print(f"[INFO] pad_token={tokenizer.pad_token} pad_token_id={tokenizer.pad_token_id}")

    # Dataset
    dataset = load_jsonl_records(args.data, default_instruction=args.default_instruction)
    split = dataset.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_ds = split["train"]
    eval_ds = split["test"]

    def tok_fn(batch):
        return tokenize_sft_batch(batch, tokenizer=tokenizer, max_length=args.max_length)

    train_ds = train_ds.map(tok_fn, batched=True, remove_columns=train_ds.column_names)
    eval_ds = eval_ds.map(tok_fn, batched=True, remove_columns=eval_ds.column_names)

    # Sanity check masking
    sanity_check_tokenization(train_ds, tokenizer, n=args.sanity_n)

    collator = DynamicSFTCollator(tokenizer=tokenizer)

    # QLoRA quant config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant_config,
        device_map=device_map,
        trust_remote_code=True,
    )

    # gradient checkpointing needs this off
    model.config.use_cache = False

    model = prepare_model_for_kbit_training(model)

    lora_targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    if not lora_targets:
        raise ValueError("Parsed lora_targets is empty; check --lora_targets.")

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_targets,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # IMPORTANT: enforce save_steps multiple of eval_steps when load_best_model_at_end=True
    eval_steps = int(args.eval_steps)
    save_steps = int(args.save_steps)
    if save_steps % eval_steps != 0:
        fixed = eval_steps * math.ceil(save_steps / eval_steps)
        print(
            f"[WARN] save_steps ({save_steps}) not a multiple of eval_steps ({eval_steps}). "
            f"Auto-fixing save_steps -> {fixed}."
        )
        save_steps = fixed

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_bs,
        per_device_eval_batch_size=args.eval_bs,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.epochs,

        bf16=args.bf16,
        fp16=args.fp16,

        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to="none",
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.log_steps,

        remove_unused_columns=False,

        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,

        dataloader_pin_memory=False,
        dataloader_num_workers=0,

        max_grad_norm=1.0,
        group_by_length=False,
        eval_accumulation_steps=1,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)],
    )

    train_result = trainer.train()
    eval_metrics = trainer.evaluate()

    final_eval_loss = eval_metrics.get("eval_loss")
    if final_eval_loss is not None:
        final_ppl = math.exp(final_eval_loss)
        print(f"[RESULT] Final Eval Loss: {final_eval_loss:.4f}")
        print(f"[RESULT] Final Perplexity: {final_ppl:.2f}")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    plot_path = os.path.join(args.output_dir, "loss_plot.png")
    plot_losses(trainer, plot_path)

    metrics_path = os.path.join(args.output_dir, "final_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"train_metrics": train_result.metrics, "eval_metrics": eval_metrics}, f, indent=2)

    print(f"[INFO] Wrote metrics to {metrics_path}")
    print(f"[DONE] Outputs saved in {args.output_dir}")


if __name__ == "__main__":
    main()

