#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Token Compression Attention for GLUE – Fully Offline & Stable
Corrected for BERT compatibility and Trainer signature.
"""

import os
import argparse
import math
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

# -------------------- Force offline (if you have local cache) --------------------
# If you have already downloaded the model, set these to True.
# Otherwise, set to False for the first run to download, then change back to True.
os.environ["HF_HUB_OFFLINE"] = "1"          # comment out for first run
os.environ["TRANSFORMERS_OFFLINE"] = "1"    # comment out for first run

# If you still get 404 errors, remove the broken endpoint:
if "HF_ENDPOINT" in os.environ and "devneeds" in os.environ["HF_ENDPOINT"]:
    del os.environ["HF_ENDPOINT"]

# ----------------------------------------------------------------------
# GLUE Utilities
# ----------------------------------------------------------------------
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

def get_task_info(task_name: str):
    t = task_name.lower()
    if t == "stsb": return 1, True, "pearson_spearman", True
    if t == "cola": return 2, False, "matthews_correlation", True
    glue_info = {
        "sst2": (2, False, "accuracy", True),
        "mrpc": (2, False, "f1", True),
        "qqp": (2, False, "f1", True),
        "mnli": (3, False, "accuracy", True),
        "qnli": (2, False, "accuracy", True),
        "rte": (2, False, "accuracy", True),
        "wnli": (2, False, "accuracy", True),
    }
    return glue_info[t]

def compute_metrics_builder(task_name):
    metric = evaluate.load("glue", task_name)
    _, is_reg, _, _ = get_task_info(task_name)
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

# ------------------------------------------------------------
# Token Compressed Attention (with correct forward indentation)
# ------------------------------------------------------------
class TokenCompressedAttention(nn.Module):
    def __init__(self, config, compressed_tokens: int, max_seq_len: int, reg_lambda: float = 0.01):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.k = compressed_tokens
        self.max_seq_len = max_seq_len
        self.reg_lambda = reg_lambda

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)

        self.compress = nn.Parameter(torch.empty(self.k, self.max_seq_len))
        nn.init.xavier_uniform_(self.compress)
        self.norm = nn.LayerNorm(self.hidden_size)

    def get_compress_matrix(self, seq_len):
        return self.compress[:, :seq_len]

    def orthogonality_loss(self):
        C = self.compress
        gram = C @ C.T
        eye = torch.eye(self.k, device=C.device)
        return F.mse_loss(gram, eye) * self.reg_lambda

    # ========== IMPORTANT: forward MUST be inside the class ==========
    def forward(self, hidden_states, attention_mask=None, head_mask=None,
                output_attentions=False, **kwargs):
        # **kwargs absorbs all extra arguments (e.g., encoder_hidden_states)
        B, N, D = hidden_states.shape
        C_mat = self.get_compress_matrix(N)                # (k, N)
        # compression: (B, N, D) * (k, N) -> (B, k, D)
        x_comp = torch.einsum("bnd,kn->bkd", hidden_states, C_mat)

        q = self.query(x_comp).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x_comp).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x_comp).view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        # Mask handling (adapt to compressed domain)
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask = attention_mask[:, 0, 0, :]   # (B, N)
            else:
                mask = attention_mask
            mask = (1.0 - mask.float()) * -1e9
            block = (N + self.k - 1) // self.k
            pad_len = (block - N % block) % block
            if pad_len > 0:
                mask = F.pad(mask, (0, pad_len), value=-1e9)
            mask = mask.view(B, self.k, block).max(dim=2).values   # (B, k)
            mask = mask.unsqueeze(1).unsqueeze(3)                   # (B,1,k,1)
            attn = attn + mask

        attn_probs = F.softmax(attn, dim=-1)
        attn_probs = F.dropout(attn_probs, p=0.1, training=self.training)
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        context = torch.matmul(attn_probs, v)                     # (B, h, k, head_dim)
        context = context.transpose(1, 2).contiguous().view(B, -1, D)  # (B, k, D)

        expand = C_mat.T                                          # (N, k)
        expanded = torch.einsum("bkd,nk->bnd", context, expand)   # (B, N, D)
        expanded = self.norm(expanded)

        if output_attentions:
            return (expanded, attn_probs)
        else:
            return (expanded, None)

# ------------------------------------------------------------
# Replace self-attention layers
# ------------------------------------------------------------
def replace_attention_with_compressor(model, compressed_tokens, max_seq_len, reg_lambda=0.01):
    for layer in model.bert.encoder.layer:
        old_attn = layer.attention.self
        new_attn = TokenCompressedAttention(model.config, compressed_tokens, max_seq_len, reg_lambda)
        new_attn.query.load_state_dict(old_attn.query.state_dict())
        new_attn.key.load_state_dict(old_attn.key.state_dict())
        new_attn.value.load_state_dict(old_attn.value.state_dict())
        layer.attention.self = new_attn
    print(f"✅ Replaced all self-attention layers (k={compressed_tokens})")
    return model

# ------------------------------------------------------------
# Custom Trainer with orthogonality regularisation
# ------------------------------------------------------------
class OrthogonalTrainer(Trainer):
    def __init__(self, reg_lambda=0.01, **kwargs):
        super().__init__(**kwargs)
        self.reg_lambda = reg_lambda

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        outputs = model(**inputs)
        loss = outputs.loss
        reg_loss = 0.0
        for module in model.modules():
            if isinstance(module, TokenCompressedAttention):
                reg_loss += module.orthogonality_loss()
        total_loss = loss + self.reg_lambda * reg_loss
        return (total_loss, outputs) if return_outputs else total_loss

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True,
                        choices=["cola","sst2","mrpc","rte","stsb","mnli","qnli","qqp","wnli"])
    parser.add_argument("--compressed_tokens", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--reg_lambda", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./compressed_attention_model")
    args = parser.parse_args()

    set_seed(args.seed)

    num_labels, is_reg, metric_name, greater_is_better = get_task_info(args.task)

    # -------------------- Load model (with fallback) --------------------
    try:
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, local_files_only=False)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config, local_files_only=False)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=False)
    except Exception as e:
        print(f"Download failed: {e}. Trying offline mode...")
        config = AutoConfig.from_pretrained("bert-base-uncased", num_labels=num_labels, local_files_only=True)
        model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", config=config, local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)

    # Replace attention layers
    model = replace_attention_with_compressor(model, args.compressed_tokens, args.max_seq_length, args.reg_lambda)

    # Load GLUE dataset (local)
    dataset = load_glue_local(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Preprocess function
    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(examples[sent1_key], truncation=True,
                                  padding='max_length', max_length=args.max_seq_length)
        else:
            tokenized = tokenizer(examples[sent1_key], examples[sent2_key], truncation=True,
                                  padding='max_length', max_length=args.max_seq_length)
        if is_reg:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized

    dataset = dataset.map(preprocess, batched=True)
    train_ds = dataset["train"]
    eval_key = "validation_matched" if args.task == "mnli" else "validation"
    eval_ds = dataset[eval_key]

    # Training arguments – use eval_strategy (not evaluation_strategy)
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        eval_strategy="epoch",          # correct keyword for newer versions
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=greater_is_better,
        save_total_limit=1,
        seed=args.seed,
        dataloader_pin_memory=False,    # MPS does not support pin_memory
    )

    compute_metrics = compute_metrics_builder(args.task)
    trainer = OrthogonalTrainer(
        reg_lambda=args.reg_lambda,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Final Evaluation: {eval_result}")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()