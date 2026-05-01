#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
GLUE Baseline — Standard BERT/DistilBERT without Token Compression
===================================================================
This is the baseline version for comparison with the compressed model.
Uses standard pretrained BERT with no modifications to attention mechanism.
"""

import os
import argparse
import logging
import warnings
from typing import Optional
from pathlib import Path
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import numpy as np
import torch
import torch.nn as nn
from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    set_seed
)
from datasets import Dataset, DatasetDict

# ── Environment ─────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── Task Configuration ──────────────────────────────────────
GLUE_TASK_KEYS = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_num_labels_and_is_regression(task_name: str):
    """Get number of labels and whether task is regression."""
    t = task_name.lower()
    if t == "stsb":
        return 1, True
    if t == "cola":
        return 2, False
    glue_num_labels = {
        "sst2": 2, "mrpc": 2, "qqp": 2, "mnli": 3,
        "qnli": 2, "rte": 2, "wnli": 2,
    }
    return glue_num_labels[t], False

# ── Data Loading ────────────────────────────────────────────
def load_glue_local(task_name: str) -> DatasetDict:
    """Load GLUE dataset from local parquet files."""
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        split = fname.split("-")[0]
        file_path = os.path.join(task_path, fname)
        splits[split] = Dataset.from_parquet([file_path])

    # Handle MNLI special splits
    if task_name == "mnli":
        for split_name in ["validation_matched", "validation_mismatched"]:
            fname = f"{split_name}.parquet"
            file_path = os.path.join(task_path, fname)
            if os.path.exists(file_path):
                splits[split_name] = Dataset.from_parquet([file_path])

    return DatasetDict(splits)

# ── Metrics ─────────────────────────────────────────────────
def get_compute_metrics(task: str):
    """Get metric computation function for task."""
    task = task.lower()
    
    if task == "cola":
        from sklearn.metrics import matthews_corrcoef
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_matthews_correlation": matthews_corrcoef(labels, preds)}
        return compute
    
    elif task == "stsb":
        from scipy.stats import pearsonr
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = logits.squeeze()
            corr, _ = pearsonr(preds, labels)
            return {"eval_pearsonr": corr}
        return compute
    
    elif task == "mrpc":
        from sklearn.metrics import accuracy_score, f1_score
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {
                "eval_accuracy": accuracy_score(labels, preds),
                "eval_f1": f1_score(labels, preds)
            }
        return compute
    
    else:
        from sklearn.metrics import accuracy_score
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_accuracy": accuracy_score(labels, preds)}
        return compute

# ── Main ────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GLUE Baseline (Standard BERT)")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./baseline_results")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    log.info("="*60)
    log.info("BASELINE MODE: Standard BERT (No Compression)")
    log.info("="*60)
    log.info(f"Task: {args.task}")
    log.info(f"Model: {args.model}")
    log.info(f"Max length: {args.max_len}")
    log.info(f"Epochs: {args.epochs}")
    log.info(f"Learning rate: {args.lr}")
    log.info(f"Batch size: {args.batch_size}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load dataset
    raw_datasets = load_glue_local(args.task)
    if 'train' not in raw_datasets:
        raise RuntimeError(f"No 'train' split in {args.task}")
    
    log.info(f"Dataset loaded: train={len(raw_datasets['train'])}, "
             f"val={len(raw_datasets.get('validation', []))}")

    # Get task info
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Tokenization
    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(
                examples[sent1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        else:
            tokenized = tokenizer(
                examples[sent1_key],
                examples[sent2_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_len,
            )
        
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        
        return tokenized

    # Apply tokenization
    dataset = raw_datasets.map(preprocess, batched=True)
    
    # Set format
    columns = ["input_ids", "attention_mask", "labels"]
    if args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
        columns.append("token_type_ids")
    dataset.set_format(type="torch", columns=columns)

    log.info(f"Tokenized: train={len(dataset['train'])}, "
             f"val={len(dataset.get('validation', []))}")

    # Load standard model (no modifications)
    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        config=config
    )

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        log.info("Gradient checkpointing enabled")

    # Device selection
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using MPS (Metal) device")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using CUDA device")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")
    
    model = model.to(device)

    # Metric for best model
    if args.task == "cola":
        metric_name = "eval_matthews_correlation"
    elif args.task == "stsb":
        metric_name = "eval_pearsonr"
    elif args.task in ("mrpc", "qqp"):
        metric_name = "eval_f1"
    else:
        metric_name = "eval_accuracy"

    # Training arguments
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        warmup_ratio=0.1,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        fp16=args.fp16,
        save_total_limit=2,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(args.task),
    )

    log.info("Starting training...")
    trainer.train()
    
    # Final evaluation
    eval_result = trainer.evaluate()
    log.info("="*60)
    log.info("BASELINE RESULTS:")
    log.info("="*60)
    for key, value in eval_result.items():
        log.info(f"{key}: {value:.4f}")
    
    # Save model
    output_path = os.path.join(args.output_dir, args.task)
    trainer.save_model(output_path)
    tokenizer.save_pretrained(output_path)
    log.info(f"Model saved to {output_path}")
    log.info("Done.")

if __name__ == "__main__":
    main()
