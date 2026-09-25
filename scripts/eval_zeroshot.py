"""
Zero-shot reasoning evaluation using LM-Evaluation-Harness protocol.
GPU-bound, batched-choice FP16 inference mode.
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

PARQUET_URLS = {
    "piqa": "https://huggingface.co/datasets/ybisk/piqa/resolve/refs%2Fconvert%2Fparquet/plain_text/validation/0000.parquet",
    "arc_easy": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Easy/test/0000.parquet",
    "arc_challenge": "https://huggingface.co/datasets/allenai/ai2_arc/resolve/refs%2Fconvert%2Fparquet/ARC-Challenge/test/0000.parquet",
    "hellaswag": "https://huggingface.co/datasets/Rowan/hellaswag/resolve/refs%2Fconvert%2Fparquet/default/validation/0000.parquet",
    "winogrande": "https://huggingface.co/datasets/allenai/winogrande/resolve/refs%2Fconvert%2Fparquet/winogrande_xl/validation/0000.parquet",
}


def score_batched_continuations(model, tokenizer, context_pairs):
    """
    Score multiple (context, continuation) options in a SINGLE batched forward pass.
    context_pairs: list of tuples [(context, continuation), ...]
    """
    device = next(model.parameters()).device
    prompts = [ctx + cont for ctx, cont in context_pairs]
    
    # Tokenize full sequences together with left padding so logits align with ends
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"
    enc = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
    input_ids = enc.input_ids.to(device)
    attention_mask = enc.attention_mask.to(device)
    tokenizer.padding_side = orig_pad_side

    # Tokenize contexts separately to know continuation start indices
    ctx_enc = tokenizer([ctx for ctx, _ in context_pairs], padding=False)
    ctx_lens = [len(ids) for ids in ctx_enc.input_ids]

    with torch.inference_mode():
        if device.type == "cuda":
            with torch.amp.autocast("cuda", dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
        else:
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

    log_probs = torch.log_softmax(logits, dim=-1)
    
    scores = []
    for idx, ctx_len in enumerate(ctx_lens):
        seq_len = attention_mask[idx].sum().item()
        if ctx_len >= seq_len:
            scores.append(-9999.0)
            continue
        # continuation tokens are at indices [ctx_len, seq_len)
        # their logprobs are predicted by logits at [ctx_len - 1, seq_len - 1)
        cont_ids = input_ids[idx, ctx_len:seq_len]
        cont_log_probs = log_probs[idx, ctx_len - 1 : seq_len - 1]
        score = cont_log_probs.gather(1, cont_ids.unsqueeze(-1)).sum().item()
        scores.append(score)

    return scores


def eval_task(model, tokenizer, task_name, num_samples=500):
    cfg = TASKS[task_name]
    try:
        if task_name in PARQUET_URLS:
            ds = load_dataset("parquet", data_files=PARQUET_URLS[task_name], split="train")
        elif "config" in cfg:
            ds = load_dataset(cfg["dataset"], cfg["config"], split=cfg["split"])
        else:
            ds = load_dataset(cfg["dataset"], split=cfg["split"])
    except Exception:
        if task_name in PARQUET_URLS:
            ds = load_dataset("parquet", data_files=PARQUET_URLS[task_name], split="train")
        else:
            raise

    correct = 0
    total = 0
    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        context = ex[cfg["input_field"]]
        
        if task_name == "winogrande":
            choices = [ex["option1"], ex["option2"]]
            label = int(ex["answer"]) - 1  # 1-indexed -> 0-indexed
            context_pairs = []
            for choice in choices:
                full = context.replace("_", choice)
                split = len(context) // 2
                context_pairs.append((full[:split], full[split:]))
        elif task_name == "piqa":
            choices = [ex["sol1"], ex["sol2"]]
            label = int(ex["label"])
            context_pairs = [(context, " " + c) for c in choices]
        elif task_name == "hellaswag":
            choices = ex["endings"]
            label = int(ex["label"])
            context_pairs = [(context, " " + c) for c in choices]
        elif "choices_field" in cfg:
            field = cfg["choices_field"]
            raw_choices = ex[field]
            if isinstance(raw_choices, dict) and "text" in raw_choices and "label" in raw_choices:
                choices = raw_choices["text"]
                labels = raw_choices["label"]
            else:
                choices = [c["text"] for c in raw_choices]
                labels = [c["label"] for c in raw_choices]
            target_key = str(ex[cfg["label_field"]]).strip()
            label = labels.index(target_key) if target_key in labels else 0
            context_pairs = [(context, " " + c) for c in choices]
        else:
            choices = [ex[f] for f in cfg.get("choices_fields", [])]
            label = int(ex[cfg["label_field"]])
            context_pairs = [(context, " " + c) for c in choices]

        # Batched scoring across all choices in a SINGLE forward pass
        scores = score_batched_continuations(model, tokenizer, context_pairs)

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
    
    if args.module != "none":
        model, _ = apply_module_to_model(model, args.module, "compressor", 
                                          rank=256, device=torch.device(device))

    if device == "cuda":
        model = model.half().to("cuda")
    else:
        model = model.to("cpu")
    model.eval()

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
    for task_clean, res in results.items():
        with open(out_dir / f"{task_clean}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)
    out_file = out_dir / f"{args.module}_zeroshot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
