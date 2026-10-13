"""
SST-2 KD Benchmark v2 — with proper best-checkpoint tracking.

Fixes:
  - Uses subset for pre-KD (fast, 200 samples)
  - Evaluates every N steps during KD
  - Keeps best checkpoint, not final
  - Detects broken modules (pre-KD < random + epsilon)
  - Detects stuck modules (no improvement over initial)
  - Produces detailed CSV + JSON report
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score

# ---- Logging setup ----
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.expanduser("~/logs/sst2_kd_v2.log")),
    ],
)
log = logging.getLogger(__name__)


# =======================================================================
#  1. Configuration
# =======================================================================

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher", default="textattack/bert-base-uncased-SST-2")
    p.add_argument("--dataset", default="glue")
    p.add_argument("--task", default="sst2")
    p.add_argument("--pre_kd_samples", type=int, default=200)
    p.add_argument("--post_kd_samples", type=int, default=872)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--kd_steps", type=int, default=200)
    p.add_argument("--kd_lr", type=float, default=1e-5)
    p.add_argument("--eval_every", type=int, default=20)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--alpha", type=float, default=0.7)
    p.add_argument("--output_dir", default=os.path.expanduser("~/Project/results/sst2_kd_v2"))
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# =======================================================================
#  2. Data loading
# =======================================================================

def load_data(args, tokenizer):
    log.info("Loading SST-2 dataset...")
    ds = load_dataset('nyu-mll/glue', args.task)
    
    def tok(ex):
        return tokenizer(ex["sentence"], truncation=True, max_length=128)
    
    ds = ds.map(tok, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    log.info(f"Train: {len(ds['train'])}, Val: {len(ds['validation'])}")
    
    # Subsets for fast eval
    val_pre = ds["validation"].select(range(min(args.pre_kd_samples, len(ds["validation"]))))
    val_full = ds["validation"].select(range(min(args.post_kd_samples, len(ds["validation"]))))
    
    collator = DataCollatorWithPadding(tokenizer)
    
    train_loader = DataLoader(
        ds["train"], batch_size=args.batch_size, shuffle=True, collate_fn=collator,
    )
    val_pre_loader = DataLoader(val_pre, batch_size=args.batch_size, collate_fn=collator)
    val_full_loader = DataLoader(val_full, batch_size=args.batch_size, collate_fn=collator)
    
    return train_loader, val_pre_loader, val_full_loader


# =======================================================================
#  3. Evaluation
# =======================================================================

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        labels.extend(batch["labels"].cpu().tolist())
    return accuracy_score(labels, preds)


# =======================================================================
#  4. KD Loss
# =======================================================================

def kd_loss(student_logits, teacher_logits, labels, T, alpha):
    """
    Standard KD: soft loss + hard CE loss.
    """
    # Soft (KD)
    soft = F.kl_div(
        F.log_softmax(student_logits / T, dim=-1),
        F.softmax(teacher_logits / T, dim=-1),
        reduction="batchmean",
    ) * (T ** 2)
    
    # Hard (CE with true labels)
    hard = F.cross_entropy(student_logits, labels)
    
    return alpha * soft + (1 - alpha) * hard


# =======================================================================
#  5. Inject attention module
# =======================================================================

def inject_module(model, module_name, device):
    """Inject attention module using unified eval patcher."""
    from neuralops.evaluation.eval_patcher import apply_module_to_model
    model, ratio = apply_module_to_model(model, module_name, rank=32, device=device)
    return model.to(device)


# =======================================================================
#  6. Single-module benchmark
# =======================================================================

def benchmark_module(
    module_name,
    teacher_model,
    tokenizer,
    train_loader,
    val_pre_loader,
    val_full_loader,
    device,
    args,
):
    log.info(f"\n{'='*70}")
    log.info(f"BENCHMARKING: {module_name}")
    log.info(f"{'='*70}")
    
    result = {
        "module": module_name,
        "status": "UNKNOWN",
        "pre_kd_acc": None,
        "best_kd_acc": None,
        "best_step": None,
        "final_kd_acc": None,
        "post_kd_acc_full": None,
        "delta_vs_pre": None,
        "delta_vs_best": None,
        "duration_s": None,
        "error": None,
    }
    
    t_start = time.time()
    
    try:
        # ---- Load student (fresh BERT) ----
        student = AutoModelForSequenceClassification.from_pretrained(
            args.teacher, num_labels=2
        ).to(device)
        
        # ---- Inject new attention ----
        student = inject_module(student, module_name, device)
        
        # ---- Pre-KD eval ----
        pre_acc = evaluate(student, val_pre_loader, device)
        result["pre_kd_acc"] = round(pre_acc * 100, 2)
        log.info(f"Pre-KD accuracy (on {args.pre_kd_samples} samples): {pre_acc*100:.2f}%")
        
        # Fast-fail: if pre-KD is broken (worse than random, < 0.30), skip KD
        if pre_acc < 0.30:
            log.warning(f"Pre-KD accuracy {pre_acc*100:.2f}% is worse than random (<30%). Marking BROKEN.")
            result["status"] = "BROKEN"
            result["duration_s"] = round(time.time() - t_start, 1)
            return result
        
        # ---- Optimizer ----
        trainable_params = [p for p in student.parameters() if p.requires_grad]
        optimizer = AdamW(
            trainable_params,
            lr=args.kd_lr,
            weight_decay=0.01,
        )
        
        # ---- Teacher in eval mode (frozen) ----
        teacher_model.eval()
        
        # ---- KD training loop ----
        student.train()
        best_acc = pre_acc
        best_state = None
        best_step = 0
        
        train_iter = iter(train_loader)
        step = 0
        
        while step < args.kd_steps:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Teacher logits
            with torch.no_grad():
                teacher_out = teacher_model(**batch)
            
            # Student forward
            student_out = student(**batch)
            
            # KD loss
            loss = kd_loss(
                student_out.logits,
                teacher_out.logits,
                batch["labels"],
                args.temperature,
                args.alpha,
            )
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()
            
            step += 1
            
            # Periodic eval
            if step % args.eval_every == 0 or step == args.kd_steps:
                acc = evaluate(student, val_pre_loader, device)
                log.info(f"  step {step}/{args.kd_steps}: loss={loss.item():.4f}, acc={acc*100:.2f}%")
                
                if acc > best_acc:
                    best_acc = acc
                    best_step = step
                    best_state = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
                    log.info(f"    ⭐ New best: {acc*100:.2f}%")
                
                student.train()
        
        result["best_kd_acc"] = round(best_acc * 100, 2)
        result["best_step"] = best_step
        
        # ---- Load best state ----
        if best_state is not None:
            student.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            log.info(f"Loaded best checkpoint from step {best_step} ({best_acc*100:.2f}%)")
        
        # ---- Final eval on full set ----
        final_acc = evaluate(student, val_full_loader, device)
        result["final_kd_acc"] = round(final_acc * 100, 2)
        result["post_kd_acc_full"] = round(final_acc * 100, 2)
        
        # ---- Delta ----
        result["delta_vs_pre"] = round((final_acc - pre_acc) * 100, 2)
        result["delta_vs_best"] = round((final_acc - pre_acc) * 100, 2)
        
        # ---- Status ----
        if final_acc > pre_acc + 0.005:
            result["status"] = "IMPROVED"
        elif abs(final_acc - pre_acc) <= 0.005:
            result["status"] = "NO_CHANGE"
            failed_file = Path(args.output_dir) / "failed_modules.txt"
            with open(failed_file, "a") as ff:
                ff.write(f"{module_name}: stuck at pre-KD accuracy ({pre_acc*100:.2f}% -> {final_acc*100:.2f}%)\n")
        else:
            result["status"] = "DEGRADED"
        
        log.info(f"Final (full {args.post_kd_samples}): {final_acc*100:.2f}%")
        log.info(f"Delta vs pre: {result['delta_vs_pre']:+.2f}%")
        log.info(f"Status: {result['status']}")
    
    except Exception as e:
        log.exception(f"Module {module_name} failed: {e}")
        result["status"] = "ERROR"
        result["error"] = str(e)[:200]
    
    result["duration_s"] = round(time.time() - t_start, 1)
    return result


# =======================================================================
#  7. Main
# =======================================================================

def main():
    args = get_args()
    torch.manual_seed(args.seed)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log.info(f"Device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.expanduser("~/logs"), exist_ok=True)
    
    # ---- Load teacher ----
    log.info(f"Loading teacher: {args.teacher}")
    tokenizer = AutoTokenizer.from_pretrained(args.teacher)
    teacher = AutoModelForSequenceClassification.from_pretrained(
        args.teacher, num_labels=2
    ).to(device).eval()
    for p in teacher.parameters():
        p.requires_grad = False
    
    # ---- Load data ----
    train_loader, val_pre_loader, val_full_loader = load_data(args, tokenizer)
    
    # ---- Teacher baseline ----
    teacher_acc = evaluate(teacher, val_full_loader, device)
    log.info(f"Teacher SST-2 accuracy (full): {teacher_acc*100:.2f}%")
    
    # ---- Get list of modules ----
    try:
        from neuralops.core.registry import ATTENTION_REGISTRY
        modules = sorted(ATTENTION_REGISTRY.keys())
    except ImportError:
        modules = [
            "adaptive_span_attention", "adaptive_sparse_attention",
            "adaptive_svd_attention", "adasplash_attention",
            "aegis_hybrid", "cosformer", "gla_attention",
        ]
    
    log.info(f"Total modules to benchmark: {len(modules)}")
    
    # ---- Benchmark each ----
    results = []
    start_time = time.time()
    
    for i, module_name in enumerate(modules, 1):
        log.info(f"\n[{i}/{len(modules)}] {module_name}")
        
        result = benchmark_module(
            module_name,
            teacher,
            tokenizer,
            train_loader,
            val_pre_loader,
            val_full_loader,
            device,
            args,
        )
        result["index"] = i
        results.append(result)
        
        # Save intermediate results
        _save_results(results, args.output_dir, teacher_acc)
        
        # Progress
        elapsed = time.time() - start_time
        avg = elapsed / i
        eta = avg * (len(modules) - i)
        log.info(f"Progress: {i}/{len(modules)}, ETA: {eta/60:.1f} min")
    
    log.info("\n" + "=" * 70)
    log.info("BENCHMARK COMPLETE")
    log.info("=" * 70)
    _print_summary(results, teacher_acc)
    _save_results(results, args.output_dir, teacher_acc)


def _save_results(results, output_dir, teacher_acc):
    """Save results to CSV and JSON."""
    import pandas as pd
    
    df = pd.DataFrame(results)
    
    # CSV
    csv_path = os.path.join(output_dir, "sst2_kd_benchmark_v2.csv")
    df.to_csv(csv_path, index=False)
    
    # JSON
    json_path = os.path.join(output_dir, "sst2_kd_benchmark_v2.json")
    with open(json_path, "w") as f:
        json.dump({
            "teacher_acc": teacher_acc,
            "results": results,
        }, f, indent=2)
    
    # Summary by status
    status_counts = {}
    for r in results:
        s = r.get("status", "UNKNOWN")
        status_counts[s] = status_counts.get(s, 0) + 1
    
    summary_path = os.path.join(output_dir, "summary_by_status.json")
    with open(summary_path, "w") as f:
        json.dump(status_counts, f, indent=2)


def _print_summary(results, teacher_acc):
    """Print a summary table."""
    print("\n" + "=" * 100)
    print(f"SST-2 KD BENCHMARK RESULTS (Teacher: {teacher_acc*100:.2f}%)")
    print("=" * 100)
    print(f"{'Module':<40} {'Pre':>6} {'Best':>6} {'Final':>6} {'Delta':>7} {'Status':<15}")
    print("-" * 100)
    
    for r in results:
        name = r["module"][:39]
        pre = f"{r['pre_kd_acc']:.2f}" if r["pre_kd_acc"] is not None else "N/A"
        best = f"{r['best_kd_acc']:.2f}" if r["best_kd_acc"] is not None else "N/A"
        final = f"{r['final_kd_acc']:.2f}" if r["final_kd_acc"] is not None else "N/A"
        delta = f"{r['delta_vs_pre']:+.2f}" if r["delta_vs_pre"] is not None else "N/A"
        status = r["status"]
        print(f"{name:<40} {pre:>6} {best:>6} {final:>6} {delta:>7} {status:<15}")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
