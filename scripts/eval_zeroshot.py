"""
Zero-shot reasoning evaluation using LM-Evaluation-Harness protocol.
High-throughput, memory-lean engine:
1. Lazy per-task dataset loading & pre-tokenization with immediate gc/empty_cache.
2. max_length=512 to cut memory buffers by 2x.
3. Default batch_size=16 for low VRAM and host RAM footprint.
4. Length-sorted batching across samples.
5. Standard Winogrande prefix/continuation split.
6. FP32 model with selective torch.amp.autocast(fp16) on CUDA.
"""
import argparse
import gc
import json
import time
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


def _forward_with_oom_fallback(model, input_ids, attention_mask, device):
    """Executes forward pass with automatic halving fallback on CUDA OOM."""
    with torch.inference_mode():
        try:
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    return model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                return model(input_ids=input_ids, attention_mask=attention_mask).logits
        except torch.cuda.OutOfMemoryError:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            half = max(1, len(input_ids) // 2)
            logits_1 = _forward_with_oom_fallback(model, input_ids[:half], attention_mask[:half], device)
            logits_2 = _forward_with_oom_fallback(model, input_ids[half:], attention_mask[half:], device)
            return torch.cat([logits_1, logits_2], dim=0)


def eval_task(model, tokenizer, task_name, num_samples=500, batch_size=64):
    cfg = TASKS[task_name]
    
    # 1. Lazy load dataset per task
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

    device = next(model.parameters()).device
    orig_pad_side = tokenizer.padding_side
    tokenizer.padding_side = "right"

    # 2. Extract only needed samples (up to num_samples)
    all_pairs = []       # list of (ctx_str, cont_str)
    all_meta = []        # list of (sample_idx, choice_idx)
    labels = []
    num_choices_per_sample = []

    for i, ex in enumerate(ds):
        if i >= num_samples:
            break
        context = ex[cfg["input_field"]]

        if task_name == "winogrande":
            choices = [ex["option1"], ex["option2"]]
            label = int(ex["answer"]) - 1
            if "_" in context:
                pre, post = context.split("_", 1)
            else:
                pre, post = context, ""
            pairs = [(pre, choice + post) for choice in choices]
        elif task_name == "piqa":
            choices = [ex["sol1"], ex["sol2"]]
            label = int(ex["label"])
            pairs = [(context, " " + c) for c in choices]
        elif task_name == "hellaswag":
            choices = ex["endings"]
            label = int(ex["label"])
            pairs = [(context, " " + c) for c in choices]
        elif "choices_field" in cfg:
            field = cfg["choices_field"]
            raw_choices = ex[field]
            if isinstance(raw_choices, dict) and "text" in raw_choices and "label" in raw_choices:
                choices = raw_choices["text"]
                choice_labels = raw_choices["label"]
            else:
                choices = [c["text"] for c in raw_choices]
                choice_labels = [c["label"] for c in raw_choices]
            target_key = str(ex[cfg["label_field"]]).strip()
            label = choice_labels.index(target_key) if target_key in choice_labels else 0
            pairs = [(context, " " + c) for c in choices]
        else:
            choices = [ex[f] for f in cfg.get("choices_fields", [])]
            label = int(ex[cfg["label_field"]])
            pairs = [(context, " " + c) for c in choices]

        labels.append(label)
        num_choices_per_sample.append(len(pairs))
        for ci, pair in enumerate(pairs):
            all_pairs.append(pair)
            all_meta.append((i, ci))

    # Free dataset object immediately
    del ds
    gc.collect()

    # 3. Pre-tokenize with max_length=512 (cuts buffer RAM/VRAM by 2x)
    all_prompts = [ctx + cont for ctx, cont in all_pairs]
    all_contexts = [ctx for ctx, _ in all_pairs]

    enc_prompts = tokenizer(all_prompts, padding=False, truncation=True, max_length=512)
    enc_contexts = tokenizer(all_contexts, padding=False, truncation=True, max_length=512)

    del all_prompts, all_contexts, all_pairs
    gc.collect()

    items = []
    for pair_idx, (p_ids, c_ids) in enumerate(zip(enc_prompts["input_ids"], enc_contexts["input_ids"])):
        c_len = min(len(c_ids), len(p_ids))
        items.append({
            "pair_idx": pair_idx,
            "meta": all_meta[pair_idx],
            "input_ids": p_ids,
            "ctx_len": c_len,
            "seq_len": len(p_ids)
        })

    del enc_prompts, enc_contexts
    gc.collect()

    # 4. Sort by length to minimize padding waste
    items.sort(key=lambda x: x["seq_len"])

    # 5. Batched scoring with batch_size=16
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    all_scores = {}
    t0 = time.time()
    n_batches = (len(items) + batch_size - 1) // batch_size

    for b_idx in range(n_batches):
        batch_items = items[b_idx * batch_size : (b_idx + 1) * batch_size]
        max_b_len = max(it["seq_len"] for it in batch_items)

        b_input_ids = []
        b_att_mask = []
        for it in batch_items:
            pad_len = max_b_len - it["seq_len"]
            b_input_ids.append(it["input_ids"] + [pad_token_id] * pad_len)
            b_att_mask.append([1] * it["seq_len"] + [0] * pad_len)

        t_input_ids = torch.tensor(b_input_ids, dtype=torch.long, device=device)
        t_att_mask = torch.tensor(b_att_mask, dtype=torch.long, device=device)

        logits = _forward_with_oom_fallback(model, t_input_ids, t_att_mask, device)
        log_probs = torch.log_softmax(logits, dim=-1)

        for row_idx, it in enumerate(batch_items):
            c_len = it["ctx_len"]
            s_len = it["seq_len"]
            if c_len >= s_len:
                score = -9999.0
            else:
                c_ids = t_input_ids[row_idx, c_len:s_len]
                c_log_probs = log_probs[row_idx, c_len - 1 : s_len - 1]
                score = c_log_probs.gather(1, c_ids.unsqueeze(-1)).sum().item()
            all_scores[it["meta"]] = score

        del logits, log_probs, t_input_ids, t_att_mask

        if (b_idx + 1) % 10 == 0 or (b_idx + 1) == n_batches:
            elapsed = time.time() - t0
            eta = (elapsed / (b_idx + 1)) * (n_batches - (b_idx + 1))
            pct = ((b_idx + 1) / n_batches) * 100
            print(f"  {task_name}: batch {b_idx + 1}/{n_batches} ({pct:.0f}%, elapsed {elapsed:.1f}s, ETA {eta:.1f}s)")

    # 6. Evaluate accuracy
    correct = 0
    total = len(labels)
    for sample_idx in range(total):
        n_c = num_choices_per_sample[sample_idx]
        sample_scores = [all_scores[(sample_idx, ci)] for ci in range(n_c)]
        pred = max(range(len(sample_scores)), key=lambda j: sample_scores[j])
        if pred == labels[sample_idx]:
            correct += 1

    tokenizer.padding_side = orig_pad_side

    # 7. Strict memory cleanup per task
    del items, all_scores, all_meta, labels, num_choices_per_sample
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    acc = correct / total * 100 if total > 0 else 0.0
    return {"task": task_name, "accuracy": round(acc, 2), "num_samples": total}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--tasks", default="piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=64)
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

    model = model.to(device).float()
    model.eval()

    results = {}
    for task in args.tasks.split(","):
        task_clean = task.strip()
        if not task_clean:
            continue
        print(f"\n=== Evaluating {task_clean} ===")
        result = eval_task(model, tokenizer, task_clean, args.num_samples, batch_size=args.batch_size)
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
