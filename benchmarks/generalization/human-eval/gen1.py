# save as gen.py and run: python gen.py --model Qwen/Qwen2.5-Coder-7B-Instruct --out samples.jsonl --n 1
import argparse, json, torch
from human_eval.data import write_jsonl, read_problems
from transformers import AutoModelForCausalLM, AutoTokenizer

def generate(model, tok, prompt, max_new_tokens=256, temperature=0.2, top_p=0.95):
    ipt = tok(prompt, return_tensors="pt").to(model.device)
    out = model.generate(**ipt, do_sample=True, temperature=temperature, top_p=top_p,
                         max_new_tokens=max_new_tokens, pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0], skip_special_tokens=True)
    # strip the original prompt from the front if the model echoes it
    return text[len(prompt):] if text.startswith(prompt) else text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="samples.jsonl")
    ap.add_argument("--n", type=int, default=1, help="samples per task")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, device_map="auto", torch_dtype=torch.float16)

    problems = read_problems()  # loads HumanEval prompts
    samples = []
    for tid, prob in problems.items():
        for _ in range(args.n):
            comp = generate(model, tok, prob["prompt"])
            samples.append({"task_id": tid, "completion": comp})
    write_jsonl(args.out, samples)

if __name__ == "__main__":
    main()

