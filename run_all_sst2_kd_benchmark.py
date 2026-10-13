#!/usr/bin/env python3
import os
import sys
import time
import json
import csv
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.evaluation.eval_patcher import apply_module_to_model

RESULTS_CSV = Path("results/sst2_kd_benchmark_summary.csv")
RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)

def evaluate_acc(model, val_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for b in val_loader:
            ids = b["input_ids"].to(device)
            mask = b["attention_mask"].to(device)
            labels = b["label"].to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds = out.logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)
    return correct / max(1, total)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INIT] Benchmark Device: {device}")
    
    teacher_name = "textattack/bert-base-uncased-SST-2"
    print(f"[INIT] Loading Tokenizer & Teacher: {teacher_name}")
    tokenizer = AutoTokenizer.from_pretrained(teacher_name)
    teacher = AutoModelForSequenceClassification.from_pretrained(teacher_name).to(device)
    teacher.eval()
    
    print("[INIT] Loading SST-2 Dataset...")
    ds = load_dataset("glue", "sst2")
    train_ds = ds["train"].select(range(min(len(ds["train"]), 1000)))
    val_ds = ds["validation"]
    
    def tokenize_fn(b):
        return tokenizer(b["sentence"], padding="max_length", truncation=True, max_length=128)
        
    train_enc = train_ds.map(tokenize_fn, batched=True)
    val_enc = val_ds.map(tokenize_fn, batched=True)
    train_enc.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    val_enc.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    
    train_loader = DataLoader(train_enc, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_enc, batch_size=32)
    
    teacher_acc = evaluate_acc(teacher, val_loader, device)
    print(f"[BASELINE] Teacher SST-2 Accuracy: {teacher_acc * 100:.2f}% (872 validation samples)")
    
    # Header for results CSV
    if not RESULTS_CSV.exists():
        with open(RESULTS_CSV, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "module", "type", "pre_acc", "post_acc", "delta_acc", "teacher_acc", "status", "duration_s"])
            
    modules_list = sorted(list(ATTENTION_REGISTRY.keys()))
    print(f"[PLAN] Starting SST-2 KD Benchmark on {len(modules_list)} attention modules...")
    
    kl_loss_fn = nn.KLDivLoss(reduction="batchmean")
    ce_loss_fn = nn.CrossEntropyLoss()
    
    for idx, mod_name in enumerate(modules_list, 1):
        print("\n" + "=" * 70)
        print(f"[{idx}/{len(modules_list)}] BENCHMARKING MODULE: {mod_name}")
        print("=" * 70)
        t0 = time.time()
        
        try:
            student = AutoModelForSequenceClassification.from_pretrained(teacher_name).to(device)
            student, ratio = apply_module_to_model(student, mod_name, "attention", rank=32, device=device)
            pre_acc = evaluate_acc(student, val_loader, device)
            print(f"  Pre-KD Accuracy : {pre_acc * 100:.2f}%")
            
            # KD Training loop (50 steps)
            trainable = [p for p in student.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable, lr=1e-4)
            
            student.train()
            step = 0
            max_steps = 50
            for epoch in range(10):
                for batch in train_loader:
                    optimizer.zero_grad()
                    ids = batch["input_ids"].to(device)
                    mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)
                    
                    with torch.no_grad():
                        t_out = teacher(input_ids=ids, attention_mask=mask)
                    s_out = student(input_ids=ids, attention_mask=mask)
                    
                    l_ce = ce_loss_fn(s_out.logits, labels)
                    l_kl = kl_loss_fn(
                        torch.log_softmax(s_out.logits / 2.0, dim=-1),
                        torch.softmax(t_out.logits / 2.0, dim=-1)
                    ) * 4.0
                    
                    loss = 0.5 * l_ce + 0.5 * l_kl
                    loss.backward()
                    optimizer.step()
                    step += 1
                    if step >= max_steps:
                        break
                if step >= max_steps:
                    break
                    
            post_acc = evaluate_acc(student, val_loader, device)
            delta_acc = post_acc - pre_acc
            dur = time.time() - t0
            print(f"  Post-KD Accuracy: {post_acc * 100:.2f}% (Delta: {delta_acc * 100:+.2f}%) in {dur:.1f}s")
            
            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mod_name, "attention", f"{pre_acc:.4f}", f"{post_acc:.4f}", f"{delta_acc:.4f}", f"{teacher_acc:.4f}", "PASS", f"{dur:.1f}"])
                
        except Exception as e:
            dur = time.time() - t0
            print(f"  ERROR on {mod_name}: {e}")
            with open(RESULTS_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), mod_name, "attention", "", "", "", f"{teacher_acc:.4f}", f"ERROR: {str(e)[:100]}", f"{dur:.1f}"])
            continue

if __name__ == "__main__":
    main()
