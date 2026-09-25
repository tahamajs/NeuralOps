"""
Zero-shot reasoning evaluation using LM-Evaluation-Harness protocol.
Evaluates models across PIQA, ARC-Easy, ARC-Challenge, HellaSwag, and WinoGrande.
"""
import argparse
import json
import logging
from pathlib import Path
import sys
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "Codes"))
from framework.eval_all_models_datasets import apply_module_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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


def score_continuation(model, tokenizer, context: str, continuation: str) -> float:
    """Compute log-prob of continuation given context."""
    if not continuation:
        return float("-inf")
    ctx_ids = tokenizer(context, return_tensors="pt").input_ids
    full_ids = tokenizer(context + continuation, return_tensors="pt").input_ids
    ctx_len = ctx_ids.shape[1]
    if full_ids.shape[1] <= ctx_len:
        return float("-inf")
    
    with torch.no_grad():
        outputs = model(full_ids.to(model.device))
        logits = outputs.logits
    
    log_probs = torch.log_softmax(logits[0, ctx_len - 1 : -1, :], dim=-1)
    cont_ids = full_ids[0, ctx_len:]
    score = log_probs.gather(1, cont_ids.unsqueeze(-1)).sum().item()
    return score


def eval_task(model, tokenizer, task_name: str, num_samples: int = 500) -> dict:
    cfg = TASKS[task_name]
    logger.info("Loading dataset for task: %s...", task_name)
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
            choices = [ex["option1"], ex["option2"]]
            label = int(ex["answer"]) - 1
            scores = []
            for choice in choices:
                full = context.replace("_", choice)
                split = len(context) // 2
                scores.append(score_continuation(model, tokenizer, full[:split], full[split:]))
        elif "choices_field" in cfg:
            raw_choices = ex[cfg["choices_field"]]
            if isinstance(raw_choices, dict):
                texts = raw_choices.get("text", [])
                keys = raw_choices.get("label", [])
                ans_key = str(ex[cfg["label_field"]])
                if ans_key in keys:
                    label = keys.index(ans_key)
                elif ans_key.isdigit() and len(texts) >= int(ans_key):
                    label = int(ans_key) - 1
                elif ans_key in ["A", "B", "C", "D"]:
                    label = ord(ans_key) - ord("A")
                else:
                    label = 0
                scores = [score_continuation(model, tokenizer, context, " " + c) for c in texts]
            elif isinstance(raw_choices, list) and len(raw_choices) > 0 and isinstance(raw_choices[0], dict):
                choices = [c["text"] for c in raw_choices]
                labels = [str(c["label"]) for c in raw_choices]
                ans_key = str(ex[cfg["label_field"]])
                if ans_key in labels:
                    label = labels.index(ans_key)
                elif ans_key.isdigit() and len(labels) >= int(ans_key):
                    label = int(ans_key) - 1
                elif ans_key in ["A", "B", "C", "D"]:
                    label = ord(ans_key) - ord("A")
                else:
                    label = 0
                scores = [score_continuation(model, tokenizer, context, " " + c) for c in choices]
            else:
                choices = raw_choices
                raw_label = ex[cfg["label_field"]]
                try:
                    label = int(raw_label)
                except ValueError:
                    label = ord(str(raw_label).upper()) - ord("A")
                scores = [score_continuation(model, tokenizer, context, " " + c) for c in choices]
        else:
            choices = [ex[f] for f in cfg["choices_fields"]]
            label = int(ex[cfg["label_field"]])
            scores = [score_continuation(model, tokenizer, context, " " + c) for c in choices]
        
        pred = max(range(len(scores)), key=lambda j: scores[j])
        if pred == label:
            correct += 1
        total += 1
        
        if (i + 1) % 100 == 0 or (i + 1) == num_samples:
            logger.info("  %s: %d/%d, running acc=%.2f%%", task_name, i + 1, num_samples, (correct / total) * 100)
    
    acc = (correct / total * 100.0) if total > 0 else 0.0
    return {"task": task_name, "accuracy": round(acc, 2), "num_samples": total, "correct": correct}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", default="none", help="Module name to apply or 'none' for dense baseline")
    ap.add_argument("--model", default="models/gpt2", help="Model path")
    ap.add_argument("--rank", type=int, default=384, help="Compression rank")
    ap.add_argument("--tasks", default="piqa,arc_easy,hellaswag,winogrande", help="Comma-separated list of tasks")
    ap.add_argument("--num_samples", type=int, default=500, help="Number of samples per task")
    ap.add_argument("--output_dir", required=True, help="Directory to save JSON results")
    args = ap.parse_args()
    
    logger.info("Loading model and tokenizer: %s...", args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(args.model).cuda()
    model.eval()
    
    if args.module != "none":
        logger.info("Applying module %s (rank=%d) to %s...", args.module, args.rank, args.model)
        model, _ = apply_module_to_model(model, args.module, "compressor", rank=args.rank, device="cuda")
        model = model.cuda()
        model.eval()
    
    results = {}
    task_list = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in task_list:
        logger.info("=== Evaluating %s ===", task)
        result = eval_task(model, tokenizer, task, args.num_samples)
        results[task] = result
        logger.info("Finished %s: accuracy = %.2f%%", task, result["accuracy"])
    
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.module}_zeroshot.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Zero-shot results successfully saved to: %s", out_path)


if __name__ == "__main__":
    main()
