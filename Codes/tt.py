#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLUE with Fixed‑Rank SVD Compression – Stable, Fast, Offline
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
from datasets import Dataset, DatasetDict
import evaluate

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

GLUE_TASK_KEYS = {
    "cola": ("sentence", None), "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def load_glue_local(task_name: str) -> DatasetDict:
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")
    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"): continue
        split = fname.split("-")[0]
        splits[split] = Dataset.from_parquet([os.path.join(task_path, fname)])
    return DatasetDict(splits)

def get_num_labels_and_is_regression(task_name: str):
    t = task_name.lower()
    if t == "stsb": return 1, True
    if t == "cola": return 2, False
    glue_num_labels = {"sst2":2, "mrpc":2, "qqp":2, "mnli":3, "qnli":2, "rte":2, "wnli":2}
    return glue_num_labels[t], False

def get_primary_metric(task_name: str):
    t = task_name.lower()
    if t == "cola": return "matthews_correlation", True
    if t == "stsb": return "pearson_spearman", True
    if t in ["mrpc", "qqp"]: return "f1", True
    return "accuracy", True

def compute_metrics_builder(task_name):
    metric = evaluate.load("glue", task_name)
    _, is_reg = get_num_labels_and_is_regression(task_name)
    def compute_metrics(p):
        preds = p.predictions
        labels = p.label_ids
        if not is_reg:
            preds = np.argmax(preds, axis=1)
        else:
            preds = preds.squeeze()
        result = metric.compute(predictions=preds, references=labels)
        if task_name == "stsb" and "pearson" in result and "spearmanr" in result:
            result["pearson_spearman"] = (result["pearson"] + result["spearmanr"]) / 2.0
        return result
    return compute_metrics

# ----------------------------------------------------------------------
# SVD Linear – merged weight for fast training
# ----------------------------------------------------------------------
class SVDLinear(nn.Module):
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

    def set_low_rank_factors(self, U, W):
        with torch.no_grad():
            self.weight.copy_(U @ W)


def compress_model_fixed_rank(model, rank=64, target_names=None, skip_classifier=True):
    if target_names is None:
        target_names = ["query", "key", "value", "dense", "intermediate", "output", "pooler"]

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if skip_classifier and "classifier" in name:
            print(f"Skipping (kept full rank): {name}")
            continue
        if not any(t in name for t in target_names):
            continue

        W_orig = module.weight.data.cpu().numpy()
        out_dim, in_dim = W_orig.shape
        effective_rank = min(rank, out_dim, in_dim)
        U, S, Vt = np.linalg.svd(W_orig, full_matrices=False)
        U_r = U[:, :effective_rank]
        W_r = np.diag(S[:effective_rank]) @ Vt[:effective_rank, :]

        svd_linear = SVDLinear(module.in_features, module.out_features,
                               effective_rank, bias=module.bias is not None)
        svd_linear.set_low_rank_factors(torch.tensor(U_r, dtype=module.weight.dtype),
                                        torch.tensor(W_r, dtype=module.weight.dtype))
        if module.bias is not None:
            svd_linear.bias.data.copy_(module.bias)

        parent = model.get_submodule('.'.join(name.split('.')[:-1])) if '.' in name else model
        setattr(parent, name.split('.')[-1], svd_linear)
        print(f"Compressed {name} to rank {effective_rank}")

    print(f"✅ Compressed to rank {rank}")
    return model


def fine_tune(model, tokenizer, task_name, output_dir, args):
    datasets = load_glue_local(task_name)
    num_labels, is_regression = get_num_labels_and_is_regression(task_name)
    sent1_key, sent2_key = GLUE_TASK_KEYS[task_name]

    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(examples[sent1_key], truncation=True,
                                  padding='max_length', max_length=args.max_seq_length)
        else:
            tokenized = tokenizer(examples[sent1_key], examples[sent2_key], truncation=True,
                                  padding='max_length', max_length=args.max_seq_length)
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized

    dataset = datasets.map(preprocess, batched=True)
    train_ds = dataset["train"]
    eval_key = "validation_matched" if task_name == "mnli" else "validation"
    eval_ds = dataset[eval_key]

    metric_name, greater_is_better = get_primary_metric(task_name)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=greater_is_better,
        save_total_limit=1,
        seed=args.seed,
        local_files_only=True,
        dataloader_pin_memory=False,
    )

    compute_metrics = compute_metrics_builder(task_name)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Fine‑tuning results: {eval_result}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["cola","sst2","mrpc","rte","stsb","mnli","qnli","qqp","wnli"])
    parser.add_argument("--rank", type=int, default=64, help="Fixed rank for SVD compression")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./svd_compressed_model")
    args = parser.parse_args()

    set_seed(args.seed)

    num_labels, _ = get_num_labels_and_is_regression(args.task)
    config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    orig_params = sum(p.numel() for p in model.parameters())
    print(f"Original parameters: {orig_params:,}")

    print(f"Compressing with fixed rank {args.rank} ...")
    model = compress_model_fixed_rank(model, rank=args.rank)

    comp_params = sum(p.numel() for p in model.parameters())
    print(f"Compressed parameters: {comp_params:,} ({comp_params/orig_params:.2%} of original)")

    print(f"Fine‑tuning on {args.task}...")
    model = fine_tune(model, tokenizer, args.task, args.output_dir, args)

    print("🎉 Done!")


if __name__ == "__main__":
    main()