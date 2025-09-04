# train_codellama_lora.py
# LoRA fine-tuning of CodeLlama (decoder-only) for bug fixing with instruction-style prompts.
#
# pip install transformers datasets accelerate peft bitsandbytes sentencepiece
import torch
import argparse, json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

PROMPT_TEMPLATE = """### Instruction:
Fix the following buggy code snippet.

### Buggy Code:
{buggy}

### Fixed Code:
"""

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            if "before" in ex and "after" in ex:
                records.append({
                    "buggy": ex["before"].strip(),
                    "fixed": ex["after"].strip(),
                })
    return Dataset.from_list(records)

def format_for_lm(example):
    prompt = PROMPT_TEMPLATE.format(buggy=example["buggy"])
    full_text = prompt + example["fixed"]
    return {"text": full_text}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--eval_jsonl", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="codellama/CodeLlama-7b-Instruct-hf")
    parser.add_argument("--out_dir", type=str, default="outputs/codellama_apr_lora")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)  # LoRA often uses slightly higher LR
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    args = parser.parse_args()

    # Load dataset
    train_ds = load_jsonl(args.train_jsonl)
    eval_ds = load_jsonl(args.eval_jsonl)
    train_ds = train_ds.map(format_for_lm, remove_columns=train_ds.column_names)
    eval_ds  = eval_ds.map(format_for_lm,  remove_columns=eval_ds.column_names)

    # Tokenizer + model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Apply LoRA config
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj","v_proj"],  # common for LLaMA/CodeLlama
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=512,
        )

    train_tok = train_ds.map(tokenize_function, batched=True, remove_columns=["text"])
    eval_tok  = eval_ds.map(tokenize_function,  batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.lr,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        fp16=True,
        report_to="none",
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=eval_tok,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(args.out_dir)
    tokenizer.save_pretrained(args.out_dir)

    print("LoRA fine-tuning completed. Adapter weights saved in:", args.out_dir)

if __name__ == "__main__":
    main()

