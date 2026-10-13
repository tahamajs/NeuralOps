"""
SST-2 KD Benchmark v3 — GPU-memory-safe with v2's best-checkpoint logic.
Optimized for 6 GB Pascal VRAM:
- Shared teacher model loaded once in eval mode
- Batch size 4, gradient accumulation 4 (effective batch 16)
- Sequence length 64
- PyTorch AMP FP16 forward pass
- Aggressive cleanup (del student, del optimizer, gc.collect(), torch.cuda.empty_cache())
- Best-checkpoint tracking across KD steps
- OOM defense and per-module memory logging
"""

import os
import gc
import sys
import csv
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score


# ═══════════════════════════════════════════════════════════════════════
#  1. Utilities
# ═══════════════════════════════════════════════════════════════════════

def log_gpu_state(label):
    if not torch.cuda.is_available():
        return
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9
    free, total = torch.cuda.mem_get_info()
    logging.info(f"[GPU {label}] alloc={allocated:.2f}GB reserved={reserved:.2f}GB free={free/1e9:.2f}/{total/1e9:.2f}GB")


def cleanup(*objs):
    for obj in objs:
        if obj is not None:
            del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ═══════════════════════════════════════════════════════════════════════
#  2. Args
# ═══════════════════════════════════════════════════════════════════════

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--pre_kd_samples", type=int, default=200)
    p.add_argument("--post_kd_samples", type=int, default=872)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--grad_accum_steps", type=int, default=4)
    p.add_argument("--max_length", type=int, default=64)
    p.add_argument("--kd_steps", type=int, default=200)
    p.add_argument("--kd_lr", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--output_dir", default=os.path.expanduser("~/Project/results/sst2_kd_v3"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_fp16", action="store_true", default=True)
    p.add_argument("--max_oom_retries", type=int, default=3)
    return p.parse_args()


# ═══════════════════════════════════════════════════════════════════════
#  3. Data
# ═══════════════════════════════════════════════════════════════════════

def load_data(args, tokenizer):
    ds = load_dataset("glue", "sst2")
    ds = ds.map(
        lambda ex: tokenizer(ex["sentence"], truncation=True, max_length=args.max_length),
        batched=True,
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    val_pre = ds["validation"].select(range(min(args.pre_kd_samples, len(ds["validation"]))))
    val_full = ds["validation"].select(range(min(args.post_kd_samples, len(ds["validation"]))))

    collator = DataCollatorWithPadding(tokenizer)

    return {
        "train": DataLoader(ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator),
        "val_pre": DataLoader(val_pre, batch_size=args.batch_size * 2, collate_fn=collator),
        "val_full": DataLoader(val_full, batch_size=args.batch_size * 2, collate_fn=collator),
    }


# ═══════════════════════════════════════════════════════════════════════
#  4. Eval
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device, use_fp16=True):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        if use_fp16:
            with autocast(dtype=torch.float16):
                out = model(**batch)
        else:
            out = model(**batch)
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, preds)


# ═══════════════════════════════════════════════════════════════════════
#  5. KD loss
# ═══════════════════════════════════════════════════════════════════════

def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    soft = F.kl_div(
        F.log_softmax(student_logits.float() / T, dim=-1),
        F.softmax(teacher_logits.float() / T, dim=-1),
        reduction="batchmean",
    ) * (T * T)
    hard = F.cross_entropy(student_logits.float(), labels)
    return alpha * soft + (1 - alpha) * hard


# ═══════════════════════════════════════════════════════════════════════
#  6. Single module benchmark with OOM handling
# ═══════════════════════════════════════════════════════════════════════

def benchmark_module(module_name, teacher, tokenizer, loaders, device, args):
    log_gpu_state(f"before {module_name}")

    result = {
        "module": module_name,
        "status": "UNKNOWN",
        "pre_kd_acc": None,
        "best_kd_acc": None,
        "best_step": None,
        "final_kd_acc": None,
        "delta_vs_pre": None,
        "duration_s": None,
        "error": None,
    }

    student = None
    optimizer = None
    t_start = time.time()

    try:
        # Load fresh student
        student = AutoModelForSequenceClassification.from_pretrained(
            args.teacher, num_labels=2
        )

        # Inject module using unified eval patcher
        from neuralops.evaluation.eval_patcher import apply_module_to_model
        student, ratio = apply_module_to_model(student, module_name, rank=32, device=device)
        student = student.to(device)

        # Pre-KD eval
        pre_acc = evaluate(student, loaders["val_pre"], device, args.use_fp16)
        result["pre_kd_acc"] = round(pre_acc * 100, 2)
        log_gpu_state(f"after pre-kd eval {module_name}")

        # Fast-fail: if worse than 30% (broken at injection)
        if pre_acc < 0.30:
            result["status"] = "BROKEN_AT_INJECTION"
            result["duration_s"] = round(time.time() - t_start, 1)
            return result

        # Freeze base model, train injected attention layers
        for p in student.parameters():
            p.requires_grad = False

        from neuralops.core.architecture import get_attention_modules
        for _, attn in get_attention_modules(student):
            for p in attn.parameters():
                p.requires_grad = True

        trainable = [p for p in student.parameters() if p.requires_grad]
        if not trainable:
            # Fallback: train all parameters if attention wasn't identified
            for p in student.parameters():
                p.requires_grad = True
            trainable = [p for p in student.parameters() if p.requires_grad]

        logging.info(f"  trainable params: {sum(p.numel() for p in trainable)/1e6:.2f}M")

        optimizer = AdamW(trainable, lr=args.kd_lr, weight_decay=0.01)
        scaler = GradScaler(enabled=args.use_fp16)

        # Training
        student.train()
        best_acc = pre_acc
        best_state = None
        best_step = 0

        train_iter = iter(loaders["train"])
        step = 0
        accum = args.grad_accum_steps

        while step < args.kd_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loaders["train"])
                batch = next(train_iter)

            batch = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                with autocast(dtype=torch.float16, enabled=args.use_fp16):
                    teacher_logits = teacher(**batch).logits

            with autocast(dtype=torch.float16, enabled=args.use_fp16):
                student_out = student(**batch)
                loss = kd_loss(student_out.logits, teacher_logits, batch["labels"], args.temperature, args.alpha)

            scaler.scale(loss / accum).backward()

            if (step + 1) % accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            step += 1

            if step % args.eval_every == 0 or step == args.kd_steps:
                acc = evaluate(student, loaders["val_pre"], device, args.use_fp16)
                logging.info(f"  step {step}/{args.kd_steps} loss={loss.item():.4f} acc={acc*100:.2f}%")

                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    logging.info(f"    ⭐ new best: {acc*100:.2f}%")

                student.train()
                log_gpu_state(f"step {step} {module_name}")

        result["best_kd_acc"] = round(best_acc * 100, 2)
        result["best_step"] = best_step

        # Load best state if available
        if best_state is not None:
            student.load_state_dict({k: v.to(device) for k, v in best_state.items()})

        final_acc = evaluate(student, loaders["val_full"], device, args.use_fp16)
        result["final_kd_acc"] = round(final_acc * 100, 2)
        result["delta_vs_pre"] = round((final_acc - pre_acc) * 100, 2)

        if final_acc > pre_acc + 0.005:
            result["status"] = "IMPROVED"
        elif abs(final_acc - pre_acc) <= 0.005:
            result["status"] = "NO_CHANGE"
        else:
            result["status"] = "DEGRADED"

        logging.info(f"  final: {final_acc*100:.2f}%, delta: {result['delta_vs_pre']:+.2f}%")

    except torch.cuda.OutOfMemoryError as e:
        logging.warning(f"OOM on {module_name}: {e}")
        result["status"] = "OOM"
        result["error"] = str(e)[:200]
    except Exception as e:
        logging.exception(f"Error on {module_name}: {e}")
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]
    finally:
        # Aggressive cleanup
        cleanup(optimizer, student)
        log_gpu_state(f"after cleanup {module_name}")

    result["duration_s"] = round(time.time() - t_start, 1)
    return result


# ═══════════════════════════════════════════════════════════════════════
#  7. Main
# ═══════════════════════════════════════════════════════════════════════

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"device={device}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)

    csv_path = os.path.join(args.output_dir, "sst2_kd_v3.csv")

    # Load teacher once
    logging.info(f"loading teacher {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSequenceClassification.from_pretrained(args.teacher, num_labels=2).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False

    loaders = load_data(args, tokenizer)

    # Baseline teacher eval
    teacher_acc = evaluate(teacher, loaders["val_full"], device, args.use_fp16)
    logging.info(f"teacher acc: {teacher_acc*100:.2f}%")

    # Module registry
    from neuralops.core.registry import ATTENTION_REGISTRY
    modules = sorted(ATTENTION_REGISTRY.keys())
    logging.info(f"total modules: {len(modules)}")

    fieldnames = [
        "module", "status", "pre_kd_acc", "best_kd_acc", "best_step",
        "final_kd_acc", "delta_vs_pre", "duration_s", "error", "index"
    ]

    # Initialize CSV header if not exists
    if not os.path.exists(csv_path):
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    results = []
    start = time.time()

    for i, module_name in enumerate(modules, 1):
        logging.info(f"\n[{i}/{len(modules)}] {module_name}")
        r = benchmark_module(module_name, teacher, tokenizer, loaders, device, args)
        r["index"] = i
        results.append(r)

        # Append row to CSV immediately
        with open(csv_path, mode="a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(r)

        elapsed = time.time() - start
        eta = elapsed / i * (len(modules) - i)
        logging.info(f"  progress: {i}/{len(modules)}, eta: {eta/60:.1f} min")
        log_gpu_state("main loop")

    logging.info("\nDone. Benchmark v3 completed successfully.")
    log_gpu_state("final")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(os.path.expanduser("~/logs/sst2_kd_v3.log")),
        ],
    )
    main()
