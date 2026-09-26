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
import os
import random
import sys
import threading
import time
import numpy as np
import psutil
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_ram_usage_gb():
    """Return (used_gb, total_gb) for process RAM (RSS) and system total."""
    process = psutil.Process(os.getpid())
    process_ram = process.memory_info().rss / 1e9
    vm = psutil.virtual_memory()
    return process_ram, vm.total / 1e9


def get_vram_usage_gb():
    """Return (used_gb, total_gb) for process GPU VRAM allocated."""
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    return allocated, total


def check_memory_limits(ram_limit_gb, vram_limit_gb, context=""):
    """Raise MemoryError if process RAM or VRAM exceeds limits."""
    ram_used, ram_total = get_ram_usage_gb()
    vram_used, vram_total = get_vram_usage_gb()
    if ram_used > ram_limit_gb:
        raise MemoryError(
            f"[{context}] Process RAM overflow: {ram_used:.2f} GB used "
            f"(limit {ram_limit_gb} GB, total {ram_total:.2f} GB)"
        )
    if vram_total > 0 and vram_used > vram_limit_gb:
        raise MemoryError(
            f"[{context}] Process VRAM overflow: {vram_used:.2f} GB used "
            f"(limit {vram_limit_gb} GB, total {vram_total:.2f} GB)"
        )
    return {"ram_used_gb": round(ram_used, 2), "vram_used_gb": round(vram_used, 2)}


def _memory_watchdog(ram_hard_limit_gb, interval=2.0):
    """Background thread: force-exit if RAM exceeds hard limit."""
    while True:
        time.sleep(interval)
        ram_used, _ = get_ram_usage_gb()
        if ram_used > ram_hard_limit_gb:
            print(f"[WATCHDOG] FATAL: RAM {ram_used:.2f} GB > {ram_hard_limit_gb} GB. Exiting.")
            os._exit(137)

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


def _forward_with_oom_fallback(model, input_ids, attention_mask, device, depth=0):
    """Forward pass with automatic halving fallback on CUDA OOM.
    
    Raises MemoryError if splitting cannot help (batch_size=1)."""
    MAX_DEPTH = 10  # safety cap
    with torch.inference_mode():
        try:
            if device.type == "cuda":
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    return model(input_ids=input_ids, attention_mask=attention_mask).logits
            else:
                return model(input_ids=input_ids, attention_mask=attention_mask).logits
        except torch.cuda.OutOfMemoryError:
            # Safety: cannot split further
            if len(input_ids) <= 1 or depth >= MAX_DEPTH:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                raise MemoryError(
                    f"VRAM OOM at batch_size={len(input_ids)}, depth={depth}. "
                    f"Cannot split further. Reduce batch_size or model size."
                )
            if device.type == "cuda":
                torch.cuda.empty_cache()
            half = len(input_ids) // 2
            logits_1 = _forward_with_oom_fallback(
                model, input_ids[:half], attention_mask[:half], device, depth + 1
            )
            logits_2 = _forward_with_oom_fallback(
                model, input_ids[half:], attention_mask[half:], device, depth + 1
            )
            return torch.cat([logits_1, logits_2], dim=0)


def eval_task(model, tokenizer, task_name, num_samples=500, batch_size=8,
              module_name="none", model_name="models/gpt2", output_dir=None,
              max_ram_gb=8.0, max_vram_gb=5.5, auto_scale_batch=True, min_batch_size=1):
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

    # 3. Pre-tokenize in chunks to avoid RAM spike
    CHUNK_SIZE = 2000  # tokenize 2000 pairs at a time
    items = []
    for chunk_start in range(0, len(all_pairs), CHUNK_SIZE):
        chunk = all_pairs[chunk_start : chunk_start + CHUNK_SIZE]
        chunk_prompts = [ctx + cont for ctx, cont in chunk]
        chunk_contexts = [ctx for ctx, _ in chunk]

        enc_p = tokenizer(chunk_prompts, padding=False, truncation=True, max_length=512)
        enc_c = tokenizer(chunk_contexts, padding=False, truncation=True, max_length=512)

        for i, (p_ids, c_ids) in enumerate(zip(enc_p["input_ids"], enc_c["input_ids"])):
            c_len = min(len(c_ids), len(p_ids))
            items.append({
                "pair_idx": chunk_start + i,
                "meta": all_meta[chunk_start + i],
                "input_ids": p_ids,
                "ctx_len": c_len,
                "seq_len": len(p_ids)
            })

        del chunk, chunk_prompts, chunk_contexts, enc_p, enc_c
        gc.collect()

    del all_pairs
    gc.collect()

    # 4. Sort by length to minimize padding waste
    items.sort(key=lambda x: x["seq_len"])

    # 5. Batched scoring with dynamic batch_size & memory checks
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    all_scores = {}
    t0 = time.time()
    current_batch_size = batch_size

    b_idx = 0
    while b_idx < len(items):
        # Check memory BEFORE each batch
        try:
            mem_stats = check_memory_limits(
                max_ram_gb, max_vram_gb, context=f"{task_name} batch {b_idx}"
            )
        except MemoryError as e:
            if auto_scale_batch and current_batch_size > min_batch_size:
                new_bs = max(min_batch_size, current_batch_size // 2)
                print(f"  [MEM] {e}. Reducing batch_size {current_batch_size} -> {new_bs}")
                current_batch_size = new_bs
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                print(f"  [ABORT] {e}")
                raise

        batch_items = items[b_idx : b_idx + current_batch_size]
        max_b_len = max(it["seq_len"] for it in batch_items)

        b_input_ids = []
        b_att_mask = []
        for it in batch_items:
            pad_len = max_b_len - it["seq_len"]
            b_input_ids.append(it["input_ids"] + [pad_token_id] * pad_len)
            b_att_mask.append([1] * it["seq_len"] + [0] * pad_len)

        t_input_ids = torch.tensor(b_input_ids, dtype=torch.long, device=device)
        t_att_mask = torch.tensor(b_att_mask, dtype=torch.long, device=device)

        try:
            logits = _forward_with_oom_fallback(model, t_input_ids, t_att_mask, device)
        except MemoryError:
            if auto_scale_batch and current_batch_size > min_batch_size:
                new_bs = max(min_batch_size, current_batch_size // 2)
                print(f"  [OOM] Reducing batch_size {current_batch_size} -> {new_bs}")
                current_batch_size = new_bs
                del t_input_ids, t_att_mask
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                continue
            else:
                raise

        # NaN safety
        if not torch.isfinite(logits).all():
            logits = torch.nan_to_num(logits, nan=-1e4, posinf=1e4, neginf=-1e4)

        # FP32 log_softmax for numerical stability
        log_probs = torch.log_softmax(logits.float(), dim=-1)

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
        b_idx += current_batch_size

        # Periodic cleanup every 5 batches
        if b_idx % (current_batch_size * 5) < current_batch_size:
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        # Progress reporting
        if b_idx % (current_batch_size * 10) < current_batch_size or b_idx >= len(items):
            elapsed = time.time() - t0
            print(f"  {task_name}: {b_idx}/{len(items)} items "
                  f"({elapsed:.1f}s, bs={current_batch_size}, "
                  f"RAM={mem_stats['ram_used_gb']}GB, "
                  f"VRAM={mem_stats['vram_used_gb']}GB)")

    # 6. Evaluate accuracy
    correct = 0
    total = len(labels)
    all_preds = []
    all_scores_per_sample = []

    for sample_idx in range(total):
        n_c = num_choices_per_sample[sample_idx]
        sample_scores = [all_scores[(sample_idx, ci)] for ci in range(n_c)]
        pred = max(range(len(sample_scores)), key=lambda j: sample_scores[j])
        all_preds.append(int(pred))
        all_scores_per_sample.append(sample_scores)
        if pred == labels[sample_idx]:
            correct += 1

    tokenizer.padding_side = orig_pad_side

    acc = correct / total * 100 if total > 0 else 0.0
    task_elapsed = round(time.time() - t0, 2)

    # Save per-sample detailed file if output_dir provided
    if output_dir:
        detailed = {
            "task": task_name,
            "module": module_name,
            "model": model_name,
            "num_samples": total,
            "accuracy": round(acc, 2),
            "per_sample": [
                {
                    "idx": i,
                    "label": int(lbl),
                    "pred": int(pred),
                    "correct": bool(lbl == pred),
                    "scores": [round(float(s), 4) for s in scores_i],
                }
                for i, (lbl, pred, scores_i) in enumerate(zip(labels, all_preds, all_scores_per_sample))
            ],
        }
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        with open(out_path / f"{task_name}_detailed.json", "w", encoding="utf-8") as f:
            json.dump(detailed, f, indent=2)

    # 7. Strict memory cleanup per task
    del items, all_scores, all_meta, labels, num_choices_per_sample, all_preds, all_scores_per_sample
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return {
        "task": task_name,
        "accuracy": round(acc, 2),
        "correct_samples": correct,
        "total_samples": total,
        "elapsed_seconds": task_elapsed,
        "batch_size": batch_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True)
    ap.add_argument("--model", default="models/gpt2")
    ap.add_argument("--tasks", default="piqa,arc_easy,arc_challenge,hellaswag,winogrande")
    ap.add_argument("--num_samples", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=8,
                    help="Initial batch size (auto-reduced on memory pressure)")
    ap.add_argument("--max_ram_gb", type=float, default=8.0,
                    help="Abort if process RAM usage exceeds this (GB)")
    ap.add_argument("--max_vram_gb", type=float, default=5.5,
                    help="Abort if process VRAM usage exceeds this (GB)")
    ap.add_argument("--auto_scale_batch", action="store_true", default=True,
                    help="Automatically reduce batch_size on memory pressure")
    ap.add_argument("--min_batch_size", type=int, default=1,
                    help="Never reduce batch_size below this")
    ap.add_argument("--enable_watchdog", action="store_true", default=True,
                    help="Force-kill if RAM exceeds max_ram_gb + 2 GB")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility")
    ap.add_argument("--output_dir", required=True)
    args = ap.parse_args()

    # Reproducibility seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Optional background watchdog
    if args.enable_watchdog and (args.max_ram_gb + 1.0 < 15.0):
        watchdog = threading.Thread(
            target=_memory_watchdog,
            args=(args.max_ram_gb + 2.0,),
            daemon=True,
        )
        watchdog.start()

    # Print memory state
    ram_used, ram_total = get_ram_usage_gb()
    vram_used, vram_total = get_vram_usage_gb()
    print(f"[START] RAM: {ram_used:.2f}/{ram_total:.2f} GB, VRAM: {vram_used:.2f}/{vram_total:.2f} GB")

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
        result = eval_task(
            model, tokenizer, task_clean,
            num_samples=args.num_samples,
            batch_size=args.batch_size,
            module_name=args.module,
            model_name=args.model,
            output_dir=args.output_dir,
            max_ram_gb=args.max_ram_gb,
            max_vram_gb=args.max_vram_gb,
            auto_scale_batch=args.auto_scale_batch,
            min_batch_size=args.min_batch_size,
        )
        results[task_clean] = result
        print(f"  {task_clean}: {result['accuracy']:.2f}%")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for task_clean, res in results.items():
        with open(out_dir / f"{task_clean}.json", "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    # Compute overall module summary
    valid_accs = [r["accuracy"] for r in results.values() if "accuracy" in r]
    avg_acc = round(sum(valid_accs) / len(valid_accs), 2) if valid_accs else None
    total_elapsed = round(sum(r.get("elapsed_seconds", 0) for r in results.values()), 2)
    
    full_report = {
        "module": args.module,
        "model": args.model,
        "device": device,
        "batch_size": args.batch_size,
        "num_samples_per_task": args.num_samples,
        "average_accuracy": avg_acc,
        "total_eval_seconds": total_elapsed,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": results
    }

    out_file = out_dir / f"{args.module}_zeroshot.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2)

    try:
        from scripts.save_metadata import save_run_metadata
        save_run_metadata(
            output_dir=args.output_dir,
            module=args.module,
            model=args.model,
            task="zeroshot_5tasks",
            extra={
                "num_samples": args.num_samples,
                "tasks_evaluated": list(results.keys()),
                "aggregate_avg": avg_acc,
            },
        )
    except Exception as e:
        print(f"Warning: could not save run_metadata.json: {e}")

    print(f"\nResults saved to {out_dir}")


if __name__ == "__main__":
    main()
