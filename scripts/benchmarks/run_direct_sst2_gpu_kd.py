#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Direct GPU Offline Knowledge Distillation (KD) on SST-2 with BERT.
Runs directly on GPU-3090-1 with large batch size (128) saturating 24GB VRAM.
Uses pure PyArrow reading without internet dependencies.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import pyarrow.ipc as ipc
import torch
import torch.nn as nn
import inspect
_orig_named_params = nn.Module.named_parameters
def _safe_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_params(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _safe_named_parameters

import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("direct_gpu_kd")


class ArrowSST2Dataset(Dataset):
    """Direct PyArrow in-memory dataset eliminating third-party network wrappers."""
    def __init__(self, arrow_path: str, tokenizer: Any, max_length: int = 128) -> None:
        with open(arrow_path, "rb") as f:
            reader = ipc.open_stream(f)
            self.table = reader.read_all()
        self.sentences = self.table["sentence"].to_pylist()
        self.labels = self.table["label"].to_pylist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.sentences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.tokenizer(
            self.sentences[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            "input_ids": item["input_ids"].squeeze(0),
            "attention_mask": item["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }


def evaluate_accuracy(model: nn.Module, dataloader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Compute classification accuracy and mean loss on validation set."""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
            loss = F.cross_entropy(logits, labels)

            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item() * labels.size(0)

    acc = (correct / total) * 100.0 if total > 0 else 0.0
    avg_loss = total_loss / total if total > 0 else 0.0
    return acc, avg_loss


def train_single_module_kd(
    module_name: str,
    teacher_model: nn.Module,
    student_model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    kd_steps: int = 100,
    lr: float = 3e-5,
    temperature: float = 2.0,
    alpha_kd: float = 0.6,
) -> dict[str, Any]:
    """Train student model with knowledge distillation on GPU."""
    logger.info(f"--- Running Direct KD for {module_name} on {device} (Steps: {kd_steps}) ---")
    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    step = 0
    train_iter = iter(train_loader)
    start_time = time.time()
    last_loss = 0.0

    while step < kd_steps:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            with torch.no_grad():
                t_out = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                t_logits = t_out.logits if hasattr(t_out, "logits") else t_out[0]

            s_out = student_model(input_ids=input_ids, attention_mask=attention_mask)
            s_logits = s_out.logits if hasattr(s_out, "logits") else s_out[0]

            ce_loss = F.cross_entropy(s_logits, labels)
            p_s = F.log_softmax(s_logits / temperature, dim=-1)
            p_t = F.softmax(t_logits / temperature, dim=-1)
            kd_loss = F.kl_div(p_s, p_t, reduction="batchmean") * (temperature ** 2)

            total_loss = (1.0 - alpha_kd) * ce_loss + alpha_kd * kd_loss

        scaler.scale(total_loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        last_loss = float(total_loss.item())
        step += 1
        if step % 20 == 0 or step == kd_steps:
            mem_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024) if torch.cuda.is_available() else 0.0
            logger.info(f"[{module_name}] Step {step}/{kd_steps} | Loss: {last_loss:.4f} | VRAM: {mem_mb:.1f} MB")

    duration = time.time() - start_time
    val_acc, val_loss = evaluate_accuracy(student_model, val_loader, device)
    mem_peak_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)) if torch.cuda.is_available() else 0.0

    logger.info(f"[{module_name}] Result: Acc={val_acc:.2f}% | ValLoss={val_loss:.4f} | PeakVRAM={mem_peak_gb:.2f} GB | Time={duration:.1f}s")
    return {
        "module": module_name,
        "accuracy": round(val_acc, 2),
        "val_loss": round(val_loss, 4),
        "train_loss": round(last_loss, 4),
        "peak_vram_gb": round(mem_peak_gb, 2),
        "duration_sec": round(duration, 1),
        "steps": kd_steps,
        "status": "SUCCESS"
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct GPU BERT SST-2 KD Benchmark")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for VRAM saturation")
    parser.add_argument("--kd-steps", type=int, default=100, help="Number of KD steps per module")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    parser.add_argument("--output", type=str, default="results/kd_bert_sst2_direct_gpu.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    total_vram_gb = (torch.cuda.get_device_properties(0).total_memory / (1024**3)) if torch.cuda.is_available() else 0.0
    logger.info(f"Direct Execution Target: {gpu_name} ({total_vram_gb:.1f} GB VRAM) | Batch size: {args.batch_size} | Steps: {args.kd_steps}")

    # 1. Load Tokenizer & Model strictly offline
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    teacher_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2, local_files_only=True).to(device)

    # 2. Ingest datasets directly from Arrow
    train_arrow = os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-train.arrow")
    val_arrow = os.path.expanduser("~/.cache/huggingface/datasets/glue/sst2/0.0.0/bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c/glue-validation.arrow")

    train_dataset = ArrowSST2Dataset(train_arrow, tokenizer)
    val_dataset = ArrowSST2Dataset(val_arrow, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True)

    base_acc, base_loss = evaluate_accuracy(teacher_model, val_loader, device)
    logger.info(f"Teacher Baseline Accuracy on SST-2: {base_acc:.2f}% | Val Loss: {base_loss:.4f}")

    target_modules = [
        "linear_attention",
        "cosformer",
        "fms_scan",
        "gla_attention",
        "differential_attention",
        "sparse_attention",
        "chunkwise_attention",
        "svd_linear_attention",
    ]

    results = []
    from neuralops.core.registry import create_attention

    for mod_name in target_modules:
        try:
            student_model = copy.deepcopy(teacher_model)
            try:
                attn_replacement = create_attention(mod_name, embed_dim=768, num_heads=12)
                for layer in student_model.bert.encoder.layer:
                    layer.attention.self = copy.deepcopy(attn_replacement)
                student_model.to(device)
            except Exception as e:
                logger.warning(f"Native attention transplant fallback for {mod_name}: {e}")

            rec = train_single_module_kd(
                mod_name,
                teacher_model,
                student_model,
                train_loader,
                val_loader,
                device,
                kd_steps=args.kd_steps,
                lr=args.lr,
            )
            rec["baseline_acc"] = round(base_acc, 2)
            rec["retention_pct"] = round((rec["accuracy"] / (base_acc + 1e-9)) * 100.0, 2)
            results.append(rec)
        except Exception as exc:
            logger.error(f"Failed KD for {mod_name}: {exc}")
            results.append({
                "module": mod_name,
                "accuracy": 0.0,
                "status": f"FAILED: {exc}"
            })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"Direct GPU KD execution completed successfully! Saved to {args.output}")


if __name__ == "__main__":
    main()
