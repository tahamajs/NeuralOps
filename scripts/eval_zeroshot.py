"""
Zero-shot reasoning evaluation using LM-Evaluation-Harness protocol.
"""
import argparse
import json
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys

# Ensure Codes directory is on sys.path
HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
CODES_DIR = PROJECT_ROOT / "Codes"
for p in [str(PROJECT_ROOT), str(CODES_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from framework.eval_all_models_datasets import apply_module_to_model


TASKS = {
    "piqa": {
        "dataset": "piqa",
        "split": "validation",
        "input_field": "goal",
        "choices_fields": ["sol1", "sol2"],
        "label_field": "label",
    },
    "arc_easy": {
        "dataset": "ai2_arc",
        "config": "ARC-Easy",
        "split": "test",
        "input_field": "question",
        "choices_field": "choices",
        "label_field": "answerKey",
    },
    "arc_challenge": {
        "dataset": "ai2_arc",
        "config": "ARC-Challenge",
        "split": "test",
        "input_field": "question",
        "choices_field": "choices",
        "label_field": "answerKey",
    },
    "hellaswag": {
        "dataset": "hellaswag",
        "split": "validation",
        "input_field": "ctx",
        "choices_field": "endings",
        "label_field": "label",
    },
    "winogrande": {
        "dataset": "winogrande",
        "config": "winogrande_xl",
        "split": "validation",
        "input_field": "sentence",
        "choices_fields": ["option1", "option2"],
        "label_field": "answer",
    },
}


def score_continuation(model, tokenizer, context, continuation):
    """Compute log-prob of continuation given context."""
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids
    ctx_len = ctx_ids.shape[1]
    
    device = next(model.parameters()).device
    with torch.no_grad():
        outputs = model(full_ids.to(device))
        logits = outputs.logits
    
    # Log-prob of continuation tokens
    log_probs = torch.log_softmax(logits[0, ctx_len-1:-1, :], dim=-1)
    cont_ids = full_ids[0, ctx_len:].to(device)
    score = log_probs.gather(1, cont_ids.unsqueeze(-1)).sum().item()
    return score


def eval_task(model, tokenizer, task_name, num_samples=500):
    cfg = TASKS[task_name]
    if "config" in cfg:
        ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
    else:
        ds = load_dataset(cfg["dataset"], split=cfg["split"])
    
    correct = 0
    total = 0
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        context = ex[cfg["input_field"]]
        if task_name == "winogrande":
            # Replace '_' with each option
            choices = [ex["option1"], ex["option2"]]
            label = int(ex["answer"]) - 1  # 1-indexed -> 0-indexed
            scores = []
            for choice in choices:
                full = context.replace("_", choice)
                split = len(context) // 2
                scores.append(score_continuation(model, tokenizer, 
                                                 full[:split], full[split:]))
        elif "choices" in cfg or "choices_field" in cfg:
            field = cfg.get("choices_field", "choices")
            raw_choices = ex[field]
            if isinstance(raw_choices, dict) and "text" in raw_choices and "label" in raw_choices:
                choices = raw_choices["text"]
                labels = raw_choices["label"]
            else:
                choices = [c["text"] for c in raw_choices]
                labels = [c["label"] for c in raw_choices]
            target_key = str(ex[cfg["label_field"]]).strip()
            label = labels.index(target_key) if target_key in labels else 0
            scores = [score_continuation(model, tokenizer, context, " " + c) 
                     for c in choices]
        else:
            # PIQA, HellaSwag: list of strings
            choices = ex[cfg["choices_field"]]
            label = int(ex[cfg["label_field"]])
            scores = [score_continuation(model, tokenizer, context, " " + c) 
                     for c in choices]
        
        pred = max(range(len(scores)), key=lambda j: scores[j])
        if pred == label:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0:
            print(f"  {task_name}: {i+1}/{num_samples}, acc={correct/total*100:.2f}%")
    
    acc = correct / total * 100 if total > 0 else 0.0
    return {"task": task_name, "accuracy": round(acc, 2), "num_samples": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--tasks", default="piqa,arc_easy,hellaswag,winogrande")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()
    
    if args.module != "none":
        model, _ = apply_module_to_model(model, args.module, "compressor", 
                                          rank=256, device=torch.device(device))
        model.to(device)
    
    results = {}
    for task in args.tasks.split(","):
        task_clean = task.strip()
        if not task_clean:
            continue
        print(f"\n=== Evaluating {task_clean} ===")
        result = eval_task(model, tokenizer, task_clean, args.num_samples)
        results[task_clean] = result
        print(f"  {task_clean}: {result['accuracy']:.2f}%")
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{args.module}_zeroshot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
