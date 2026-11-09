#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Fine-tune canonical BERT-base teacher model on GLUE SST-2 with sequence classification head.

Produces the authoritative teacher checkpoint with high accuracy (~92%) for
meaningful comparative distillation and downstream evaluation across all
131 Attention Mechanisms and 82 Compression Engines.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Dynamic monkey-patch for PyTorch 1.12 compatibility with modern accelerate
_orig_named_parameters = nn.Module.named_parameters
def _compat_named_parameters(self, prefix="", recurse=True, remove_duplicate=True):
    return _orig_named_parameters(self, prefix=prefix, recurse=recurse)
nn.Module.named_parameters = _compat_named_parameters

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_bert_teacher_sst2")


def evaluate_sst2(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate accuracy and loss on GLUE SST-2 validation loader."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                loss = criterion(logits, labels)

            total_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(labels)

    avg_loss = total_loss / max(total, 1)
    acc = (correct / max(total, 1)) * 100.0
    return acc, avg_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune BERT Teacher on GLUE SST-2")
    parser.add_argument("--model_name", default="bert-base-uncased")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_seq_len", type=int, default=128)
    parser.add_argument("--save_dir", default="models/bert_sst2_teacher")
    parser.add_argument("--metrics_out", default="results/bert_attention_zoo/teacher_sst2_metrics.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    logger.info("Initializing BERT Teacher Fine-Tuning on GLUE SST-2 (Device: %s)", device)

    # Offline safety
    os.environ["HF_HUB_OFFLINE"] = "1"

    from transformers import AutoTokenizer, BertForSequenceClassification
    import datasets

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2, local_files_only=True)
    model.to(device)

    logger.info("Loading GLUE SST-2 dataset from cache...")
    raw_dataset = datasets.load_dataset("glue", "sst2")

    def tokenize_fn(examples):
        return tokenizer(examples["sentence"], truncation=True, max_length=args.max_seq_len, padding="max_length")

    logger.info("Tokenizing GLUE SST-2 splits...")
    tokenized_train = raw_dataset["train"].map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])
    tokenized_val = raw_dataset["validation"].map(tokenize_fn, batched=True, remove_columns=["sentence", "idx"])

    tokenized_train.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    tokenized_val.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(tokenized_train, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(tokenized_val, batch_size=args.batch_size, shuffle=False)

    # Initial zero-shot accuracy before training head
    initial_acc, initial_loss = evaluate_sst2(model, val_loader, device)
    logger.info("Initial Un-finetuned Head SST-2 Accuracy: %.2f%% (Loss: %.4f)", initial_acc, initial_loss)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        steps = 0

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            steps += 1

            if step % 50 == 0:
                logger.info("Epoch %d | Step %d/%d | Loss: %.4f", epoch, step, len(train_loader), loss.item())

        val_acc, val_loss = evaluate_sst2(model, val_loader, device)
        logger.info("=== Epoch %d Evaluation: Val Acc = %.2f%% | Val Loss = %.4f ===", epoch, val_acc, val_loss)
        if val_acc > best_acc:
            best_acc = val_acc
            save_path = Path(args.save_dir)
            save_path.mkdir(parents=True, exist_ok=True)
            logger.info("Saving best model to %s (Acc: %.2f%%)", save_path, val_acc)
            model.save_pretrained(save_path)
            tokenizer.save_pretrained(save_path)

    elapsed = time.time() - t0
    logger.info("Fine-Tuning Complete in %.1fs. Best SST-2 Accuracy: %.2f%%", elapsed, best_acc)

    # Write metrics artifact
    metrics_record = {
        "model": args.model_name,
        "task": "glue_sst2",
        "initial_untrained_acc": round(initial_acc, 2),
        "fine_tuned_teacher_acc": round(best_acc, 2),
        "validation_samples": len(tokenized_val),
        "train_samples": len(tokenized_train),
        "epochs": args.epochs,
        "training_time_secs": round(elapsed, 2),
        "device": str(device),
        "checkpoint_path": str(save_path.resolve()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    out_file = Path(args.metrics_out)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(metrics_record, indent=2), encoding="utf-8")
    logger.info("Saved teacher metrics to %s", out_file)


if __name__ == "__main__":
    main()
