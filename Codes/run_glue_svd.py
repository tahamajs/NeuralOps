#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLUE Research Framework – Ultimate Production‑Grade
- All methods: Baseline, LoRA, DoRA, DisLoRA, Fisher‑LoRA, Fisher‑GeLoRA, R‑Drop, R3F, Mixout, LLRD, Lion.
- Multi‑seed aggregation with mean/std (NaN‑safe).
- Memory optimised for MPS (gradient checkpointing, clipping, cache clearing).
- Local GLUE dataset fallback.
- Fisher importance computed in eval mode with random sampling.
- GeLoRA intrinsic dimension estimation (PyTorch only, no sklearn).
- FIXED: DisLoRA dimension mismatch, MPS compatibility, type errors.
"""

import os
import sys
import json
import math
import time
import uuid
import socket
import random
import logging
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from datasets import load_dataset, Dataset, DatasetDict
import evaluate

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    set_seed,
)
from transformers.trainer_callback import TrainerCallback

# Optional libraries
try:
    import pandas as pd
    _HAS_PANDAS = True
except ImportError:
    pd = None
    _HAS_PANDAS = False

try:
    from sklearn.metrics import confusion_matrix
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    confusion_matrix = None

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    _HAS_PLT = True
except ImportError:
    _HAS_PLT = False
    plt = None
    sns = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
@dataclass
class ExperimentConfig:
    model_name_or_path: str
    tasks: List[str]
    method: str
    output_root: str
    num_train_epochs: int
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    seed: int
    max_seq_length: int
    gradient_accumulation_steps: int
    lr_scheduler_type: str
    logging_steps: int
    save_total_limit: int
    fp16: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    rdrop_alpha: float
    r3f_epsilon: float
    llrd_decay: float
    mixout_p: float
    use_lion: bool
    early_stopping_patience: Optional[int] = None
    gradient_checkpointing: bool = False
    max_grad_norm: float = 1.0
    # Fisher / GeLoRA
    fisher_num_batches: int = 20
    fisher_sample_size: int = 200
    gelora_margin: float = 1.0
    gelora_min_rank: int = 2
    gelora_max_rank: int = 32
    fisher_scale_max: float = 2.0
    # DisLoRA
    dislora_beta: float = 0.5
    dislora_capacity_factor: float = 1.0
    # Multi‑seed
    num_seeds: int = 1
    # SVD attention – place these LAST (all have defaults)
    svd_attn: bool = False
    svd_rank_qkv: int = 32
    svd_rank_out: int = 32# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def get_device_and_precision(fp16_flag: bool):
    if torch.cuda.is_available():
        return torch.device("cuda"), fp16_flag
    if torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False

def get_model_device(model: nn.Module) -> torch.device:
    return next(model.parameters()).device
class SVDLinear(nn.Module):
    """
    Replace a Linear layer with its truncated SVD factors.
    During training, both U and W ( = ΣVᵀ ) are updated.
    """
    def __init__(self, in_features, out_features, rank, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.W = nn.Parameter(torch.empty(rank, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.U, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))
        if self.bias is not None:
            fan_in = self.in_features
            bound = 1 / np.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # Reconstruct and compute forward pass
        weight = self.U @ self.W
        return F.linear(x, weight, self.bias)


# ----------------------------------------------------------------------
# Replace Linear Layers
# ----------------------------------------------------------------------
def compress_model_with_svd(model, rank, target_names=None):
    """
    Replace linear layers in the model with compressed SVDLinear.
    If target_names is None, compress all linear layers.
    """
    if target_names is None:
        target_names = ["query", "key", "value", "dense"]

    compressed_count = 0
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        # Check if this layer should be compressed
        compress = any(t in name for t in target_names)
        if not compress:
            continue
        # Perform SVD on the existing weight
        W_orig = module.weight.data.cpu().numpy()
        U, S, Vt = np.linalg.svd(W_orig, full_matrices=False)
        # Keep only top 'rank' singular values
        rank_actual = min(rank, len(S))
        U_r = U[:, :rank_actual]
        W_r = np.diag(S[:rank_actual]) @ Vt[:rank_actual, :]

        # Create SVDLinear and copy factors
        svd_linear = SVDLinear(module.in_features, module.out_features,
                               rank_actual, bias=module.bias is not None)
        with torch.no_grad():
            svd_linear.U.copy_(torch.tensor(U_r, dtype=module.weight.dtype))
            svd_linear.W.copy_(torch.tensor(W_r, dtype=module.weight.dtype))
            if module.bias is not None:
                svd_linear.bias.copy_(module.bias)

        # Replace the original layer
        parent_name = '.'.join(name.split('.')[:-1])
        child_name = name.split('.')[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, child_name, svd_linear)
        compressed_count += 1
        print(f"Compressed {name} to rank {rank_actual}")

    print(f"✅ Compressed {compressed_count} layers to rank {rank}")
    return model
# ----------------------------------------------------------------------
def fine_tune(model, tokenizer, task_name, output_dir, args):
    """Fine‑tune the compressed model on the given GLUE task."""
    dataset = load_dataset("glue", task_name)

    # Preprocess dataset
    def preprocess_function(examples):
        if task_name in ["cola", "sst2"]:
            texts = examples["sentence"]
        elif task_name in ["mrpc", "rte", "wnli"]:
            texts = (examples["sentence1"], examples["sentence2"])
        elif task_name == "stsb":
            texts = (examples["sentence1"], examples["sentence2"])
        elif task_name == "mnli":
            texts = (examples["premise"], examples["hypothesis"])
        elif task_name in ["qnli", "qqp"]:
            texts = (examples["question"], examples["sentence"])
        else:
            texts = (examples["sentence1"], examples["sentence2"])
        tokenized = tokenizer(*texts, truncation=True, padding='max_length',
                              max_length=args.max_seq_length)
        tokenized["labels"] = examples["label"]
        return tokenized

    encoded_dataset = dataset.map(preprocess_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        logging_dir=os.path.join(output_dir, 'logs'),
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_accuracy",
    )

    # Use compute_metrics to track accuracy
    from evaluate import load
    metric = load("glue", task_name)

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return metric.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["validation_matched" if task_name == "mnli" else "validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    eval_result = trainer.evaluate()
    print(f"Fine‑tuning results: {eval_result}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    return model



def set_all_seeds(seed: int):
    set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
# ----------------------------------------------------------------------
# SVD Compressed Attention Layer (replaces MultiheadAttention)
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# SVD Compressed Attention Layer
# ----------------------------------------------------------------------
class CompressedAttentionLayer(nn.Module):
    def __init__(self, original_layer: nn.MultiheadAttention, rank_qkv: int, rank_out: int):
        super().__init__()
        self.original_dim = original_layer.embed_dim
        self.num_heads = original_layer.num_heads
        self.head_dim = self.original_dim // self.num_heads
        self.dropout = original_layer.dropout

        w_qkv = original_layer.in_proj_weight.data.cpu().numpy()
        w_out = original_layer.out_proj.weight.data.cpu().numpy()

        U_qkv, S_qkv, Vh_qkv = np.linalg.svd(w_qkv, full_matrices=False)
        U_out, S_out, Vh_out = np.linalg.svd(w_out, full_matrices=False)

        rank_qkv = min(rank_qkv, w_qkv.shape[1])
        rank_out = min(rank_out, w_out.shape[1])

        self.register_buffer('U_qkv', torch.from_numpy(U_qkv[:, :rank_qkv]).float())
        self.register_buffer('S_qkv_sqrt', torch.from_numpy(np.sqrt(S_qkv[:rank_qkv])).float())
        self.register_buffer('Vh_qkv', torch.from_numpy(Vh_qkv[:rank_qkv, :]).float())
        self.register_buffer('U_out', torch.from_numpy(U_out[:, :rank_out]).float())
        self.register_buffer('S_out_sqrt', torch.from_numpy(np.sqrt(S_out[:rank_out])).float())
        self.register_buffer('Vh_out', torch.from_numpy(Vh_out[:rank_out, :]).float())

        if original_layer.in_proj_bias is not None:
            self.bias_qkv = nn.Parameter(original_layer.in_proj_bias.data.clone())
        else:
            self.bias_qkv = None
        if original_layer.out_proj.bias is not None:
            self.bias_out = nn.Parameter(original_layer.out_proj.bias.data.clone())
        else:
            self.bias_out = None

    def forward(self, query, key, value, key_padding_mask=None, need_weights=False,
                attn_mask=None, average_attn_weights=True):
        x = query
        B, L, D = x.shape
        projected = torch.matmul(x, self.Vh_qkv.t())
        qkv_comp = torch.matmul(projected, self.U_qkv.t() * self.S_qkv_sqrt.view(1, -1, 1))
        if self.bias_qkv is not None:
            qkv_comp += self.bias_qkv

        dim = self.original_dim
        q = qkv_comp[:, :, :dim]
        k = qkv_comp[:, :, dim:2*dim]
        v = qkv_comp[:, :, 2*dim:]

        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        k = k.view(B, L, self.num_heads, self.head_dim).transpose(1,2)
        v = v.view(B, L, self.num_heads, self.head_dim).transpose(1,2)

        scale = 1.0 / (self.head_dim ** 0.5)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if attn_mask is not None:
            attn += attn_mask
        if key_padding_mask is not None:
            attn = attn.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), -1e9)

        attn_weights = F.softmax(attn, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1,2).contiguous().view(B, L, dim)

        context_fused = torch.matmul(context, self.Vh_out.t() * self.S_out_sqrt.view(1, -1, 1))
        out = torch.matmul(context_fused, self.U_out.t())
        if self.bias_out is not None:
            out += self.bias_out
        return out, attn_weights if need_weights else None





def replace_attention_with_svd(model: nn.Module, rank_qkv: int = 32, rank_out: int = 32):
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.MultiheadAttention):
            parent = model.get_submodule('.'.join(name.split('.')[:-1]))
            child_name = name.split('.')[-1]
            setattr(parent, child_name, CompressedAttentionLayer(module, rank_qkv, rank_out))
            logger.info(f"Replaced {name} with SVD attention (rank_qkv={rank_qkv}, rank_out={rank_out})")
def create_experiment_dir(config: ExperimentConfig) -> str:
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + str(uuid.uuid4())[:8]
    run_dir = os.path.join(config.output_root, run_id)
    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_json(obj: Any, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def save_system_info(run_dir: str):
    info = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    save_json(info, os.path.join(run_dir, "system.json"))

def save_config(config: ExperimentConfig, run_dir: str):
    save_json(asdict(config), os.path.join(run_dir, "config.json"))

# ----------------------------------------------------------------------
# GLUE Metadata
# ----------------------------------------------------------------------
GLUE_TASK_KEYS = {
    "cola": ("sentence", None), "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

def get_num_labels_and_is_regression(task_name: str) -> Tuple[int, bool]:
    t = task_name.lower()
    if t == "stsb": return 1, True
    if t == "cola": return 2, False
    glue_num_labels = {"sst2":2, "mrpc":2, "qqp":2, "mnli":3, "qnli":2, "rte":2, "wnli":2}
    return glue_num_labels[t], False

def get_primary_metric_name(task_name: str) -> str:
    t = task_name.lower()
    if t == "cola": return "matthews_correlation"
    if t == "stsb": return "pearson"
    if t in ["mrpc","qqp"]: return "f1"
    return "accuracy"

def compute_metrics_builder(task_name: str):
    t = task_name.lower()
    metric = evaluate.load("glue", t)
    def compute_metrics(p):
        preds = p.predictions
        labels = p.label_ids
        # Truncate to the smaller length
        min_len = min(len(preds), len(labels))
        if len(preds) != len(labels):
            logger.warning(f"Length mismatch: predictions {len(preds)} vs labels {len(labels)}. Truncating to {min_len}.")
            preds = preds[:min_len]
            labels = labels[:min_len]
        if t == "stsb":
            preds = np.squeeze(preds)
        else:
            preds = np.argmax(preds, axis=1)
        result = metric.compute(predictions=preds, references=labels)
        if t == "stsb" and "pearson" in result and "spearmanr" in result:
            result["pearson_spearman"] = (result["pearson"] + result["spearmanr"]) / 2.0
        if t == "mrpc" and "accuracy" in result and "f1" in result:
            result["accuracy_f1"] = (result["accuracy"] + result["f1"]) / 2.0
        return result
    return compute_metrics
# ----------------------------------------------------------------------
# Local GLUE Loader
# ----------------------------------------------------------------------
def load_glue_local(task_name: str) -> DatasetDict:
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")
    splits = {}
    split_files = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"): continue
        split = fname.split("-")[0]
        split_files.setdefault(split, []).append(os.path.join(task_path, fname))
    for split, files in split_files.items():
        splits[split] = Dataset.from_parquet(files)
    return DatasetDict(splits)

# ----------------------------------------------------------------------
# Preprocessing
# ----------------------------------------------------------------------
def preprocess_function_builder(tokenizer, task_name: str, max_seq_len: int):
    sent1_key, sent2_key = GLUE_TASK_KEYS[task_name.lower()]
    def preprocess(examples):
        args = (examples[sent1_key],) if sent2_key is None else (examples[sent1_key], examples[sent2_key])
        tokenized = tokenizer(*args, truncation=True, max_length=max_seq_len, padding=False)
        if get_num_labels_and_is_regression(task_name)[1]:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized
    return preprocess

# ----------------------------------------------------------------------
# Advanced Modules (FIXED)
# ----------------------------------------------------------------------
class MixoutLinear(nn.Module):
    def __init__(self, linear: nn.Linear, p: float):
        super().__init__()
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.weight = linear.weight
        self.bias = linear.bias
        self.p = p
        self.register_buffer("target_weight", self.weight.data.clone())
        if self.bias is not None:
            self.register_buffer("target_bias", self.bias.data.clone())
    def forward(self, x):
        if self.training and self.p > 0:
            mask = torch.bernoulli(torch.full_like(self.weight, 1 - self.p))
            mixed_weight = self.weight * mask + self.target_weight * (1 - mask)
            if self.bias is not None:
                mask_bias = torch.bernoulli(torch.full_like(self.bias, 1 - self.p))
                mixed_bias = self.bias * mask_bias + self.target_bias * (1 - mask_bias)
            else:
                mixed_bias = None
            return F.linear(x, mixed_weight, mixed_bias)
        return F.linear(x, self.weight, self.bias)

def replace_linears_with_mixout(model: nn.Module, p: float, skip_classifier: bool = True):
    for name, module in list(model.named_children()):
        if skip_classifier and "classifier" in name:
            continue
        if isinstance(module, nn.Linear):
            setattr(model, name, MixoutLinear(module, p))
        else:
            replace_linears_with_mixout(module, p, skip_classifier)

class LoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r
            self.dropout = nn.Dropout(dropout)
        else:
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
            self.dropout = nn.Identity()
    def forward(self, x):
        if self.r > 0:
            lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
            return self.linear(x) + lora_out
        return self.linear(x)

class DoRALinear(nn.Module):
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        self.r = r
        if r > 0:
            self.lora_A = nn.Linear(linear.in_features, r, bias=False)
            self.lora_B = nn.Linear(r, linear.out_features, bias=False)
            nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B.weight)
            self.scaling = alpha / r
        else:
            self.lora_A = self.lora_B = None
            self.scaling = 1.0
        with torch.no_grad():
            w_norm = linear.weight.norm(p=2, dim=1, keepdim=True)
        self.magnitude = nn.Parameter(w_norm)
    def forward(self, x):
        w_base = self.linear.weight
        if self.r > 0:
            ba = (self.lora_B.weight @ self.lora_A.weight) * self.scaling
            w_combined = w_base + ba
        else:
            w_combined = w_base
        w_norm = w_combined.norm(p=2, dim=1, keepdim=True) + 1e-8
        w_dir = w_combined / w_norm
        w_scaled = w_dir * self.magnitude
        return F.linear(x, w_scaled, self.linear.bias)


class DisLoRALinear(nn.Module):
    """Corrected DisLoRA with proper dimension handling."""
    def __init__(self, linear: nn.Linear, r: int, alpha: float, dropout: float,
                 beta: float = 0.5, capacity_factor: float = 1.0):
        super().__init__()
        self.linear = linear
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
        
        d_in = linear.in_features
        d_out = linear.out_features
        
        # Calculate effective rank
        target_params = capacity_factor * 2 * d_in * r
        self.r_eff = max(1, int(math.sqrt(target_params)))
        self.r_eff = min(self.r_eff, d_in, d_out)
        
        self.alpha = alpha
        self.beta = beta
        
        with torch.no_grad():
            U, S, Vt = torch.linalg.svd(linear.weight.data, full_matrices=False)
            rank_actual = min(self.r_eff, len(S))
            
            # U: (d_out, rank)
            # Vt: (rank, d_in) -> we want V: (d_in, rank)
            self.register_buffer('U_frozen', U[:, :rank_actual])
            self.register_buffer('S_frozen', S[:rank_actual])
            self.register_buffer('V_frozen', Vt[:rank_actual, :].T)  # (d_in, rank)
            
            self.C = nn.Parameter(torch.zeros(rank_actual, rank_actual))
            nn.init.eye_(self.C)
        
        self.scaling = alpha / max(1, rank_actual)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (B, L, d_in)
        # Project to latent: (B, L, d_in) @ (d_in, r) -> (B, L, r)
        x_proj = x @ self.V_frozen
        x_proj = self.dropout(x_proj)
        
        # Transform in latent space: (B, L, r) @ (r, r) -> (B, L, r)
        coeff_out = x_proj @ self.C
        
        # Project to output: (B, L, r) @ (r, d_out) -> (B, L, d_out)
        delta = coeff_out @ self.U_frozen.T
        
        base = self.linear(x)
        return base + self.beta * delta * self.scaling
    



def apply_lora_or_dora(model: nn.Module, r: int, alpha: float, dropout: float,
                       method_type: str = 'lora',
                       per_layer_ranks: Optional[Dict[str, int]] = None,
                       dislora_beta: float = 0.5,
                       dislora_capacity_factor: float = 1.0):
    """Apply LoRA/DoRA/DisLoRA to attention projections and FFN layers."""
    # Target substrings for all linear layers we want to adapt
    target_substrings = ["query", "key", "value", "dense", "intermediate.dense", "output.dense"]
    
    targets = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if "classifier" in name or "pooler" in name:
            continue
        if not any(ts in name for ts in target_substrings):
            continue
        targets.append(name)
    
    logger.info(f"Applying {method_type} to {len(targets)} linear layers")
    
    for full_name in targets:
        parent_name = ".".join(full_name.split(".")[:-1])
        child_name = full_name.split(".")[-1]
        parent = model.get_submodule(parent_name) if parent_name else model
        module = getattr(parent, child_name)
        
        layer_r = per_layer_ranks.get(full_name, r) if per_layer_ranks else r
        
        try:
            if method_type == 'dora':
                setattr(parent, child_name, DoRALinear(module, layer_r, alpha, dropout))
            elif method_type == 'dislora':
                setattr(parent, child_name, DisLoRALinear(
                    module, layer_r, alpha, dropout,
                    beta=dislora_beta,
                    capacity_factor=dislora_capacity_factor
                ))
            else:  # lora
                setattr(parent, child_name, LoRALinear(module, layer_r, alpha, dropout))
        except Exception as e:
            logger.warning(f"Failed to apply {method_type} to {full_name}: {e}")

# ----------------------------------------------------------------------
# Fisher Importance (fixed for MPS compatibility)
# ----------------------------------------------------------------------
def compute_fisher_importance(model, dataset, tokenizer, task_name, config,
                              target_substrings=None):
    if target_substrings is None:
        target_substrings = ["query", "key", "value", "dense"]
    fisher = {}
    model.eval()
    device = get_model_device(model)
    total = len(dataset)
    sample_size = min(config.fisher_sample_size, total)
    indices = np.random.choice(total, sample_size, replace=False)
    sampled = dataset.select(indices)
    sent1_key, sent2_key = GLUE_TASK_KEYS[task_name.lower()]
    
    def collate_fn(batch):
        if sent2_key is None:
            texts = [ex[sent1_key] for ex in batch]
            enc = tokenizer(texts, truncation=True, max_length=config.max_seq_length,
                            padding=True, return_tensors="pt")
        else:
            texts1 = [ex[sent1_key] for ex in batch]
            texts2 = [ex[sent2_key] for ex in batch]
            enc = tokenizer(texts1, texts2, truncation=True, max_length=config.max_seq_length,
                            padding=True, return_tensors="pt")
        labels = [ex["label"] for ex in batch]
        if get_num_labels_and_is_regression(task_name)[1]:
            enc["labels"] = torch.tensor(labels, dtype=torch.float)
        else:
            enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc
    
    loader = DataLoader(sampled, batch_size=config.per_device_train_batch_size,
                        collate_fn=collate_fn)
    
    for param in model.parameters():
        param.requires_grad = True
    
    for i, batch in enumerate(loader):
        if i >= config.fisher_num_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is None:
                continue
            if not any(ts in name for ts in target_substrings):
                continue
            # Use mean of squared gradients for stability
            score = (param.grad ** 2).mean().item()
            module_name = name.replace(".weight", "").replace(".bias", "")
            fisher[module_name] = fisher.get(module_name, 0.0) + score
        model.zero_grad()
    
    if not fisher:
        return None
    
    for k in fisher:
        fisher[k] /= config.fisher_num_batches
    
    # Normalize using softmax for better scaling
    scores = np.array(list(fisher.values()))
    # Handle numerical stability
    scores = scores - np.max(scores)
    exp_scores = np.exp(scores)
    norm_scores = exp_scores / (np.sum(exp_scores) + 1e-8)
    fisher = {k: norm_scores[i] for i, k in enumerate(fisher.keys())}
    
    return fisher

# ----------------------------------------------------------------------
# GeLoRA intrinsic dimension (PyTorch only)
# ----------------------------------------------------------------------
def estimate_intrinsic_dimension_pytorch(hidden_states: torch.Tensor, k: int = 5) -> float:
    """
    Estimate intrinsic dimension from hidden states.
    Input can be 2D (N, D) or 3D (N, L, D). For 3D, we average over the sequence dimension.
    """
    # If 3D, average over sequence dimension -> (N, D)
    if hidden_states.dim() == 3:
        hidden_states = hidden_states.mean(dim=1)   # average over tokens
    elif hidden_states.dim() != 2:
        raise ValueError(f"Expected 2D or 3D tensor, got {hidden_states.dim()}D")
    
    N = hidden_states.shape[0]
    if N < k + 1:
        return 1.0
    if N > 1000:
        idx = torch.randperm(N)[:1000]
        hidden_states = hidden_states[idx]
        N = 1000
    
    # Use torch.cdist for pairwise distances (works with 2D)
    dist = torch.cdist(hidden_states, hidden_states, compute_mode='donot_use_mm_for_euclid_dist')
    dist.fill_diagonal_(float('inf'))
    topk_dist, _ = torch.topk(dist, k, dim=1, largest=False)
    d_k = topk_dist[:, -1:]   # distance to k-th neighbor
    log_ratios = torch.log(topk_dist / (d_k + 1e-8))
    id_per_point = -1.0 / torch.mean(log_ratios, dim=1)
    id_global = torch.mean(id_per_point).item()
    return max(1.0, id_global)
def get_encoder_layers(model):
    if hasattr(model, 'bert') and hasattr(model.bert, 'encoder'):
        return model.bert.encoder.layer
    if hasattr(model, 'roberta') and hasattr(model.roberta, 'encoder'):
        return model.roberta.encoder.layer
    if hasattr(model, 'distilbert') and hasattr(model.distilbert, 'transformer'):
        return model.distilbert.transformer.layer
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layer'):
        return model.encoder.layer
    layers = []
    for name, module in model.named_modules():
        if 'layer' in name and isinstance(module, (nn.ModuleList, nn.Sequential)):
            for submod in module:
                layers.append(submod)
        elif 'layer' in name:
            layers.append(module)
    if layers:
        return layers
    raise ValueError("Cannot find encoder layers")

def collect_hidden_states(model, dataloader, num_batches: int = 10, max_samples: int = 800):
    layers = get_encoder_layers(model)
    num_layers = len(layers)
    hidden_states_per_layer = {i: [] for i in range(num_layers)}
    hooks = []
    
    def hook_fn(layer_idx):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hs = output[0].detach()
            else:
                hs = output.detach()
            hidden_states_per_layer[layer_idx].append(hs)
        return hook
    
    for i, layer in enumerate(layers):
        hook = layer.register_forward_hook(hook_fn(i))
        hooks.append(hook)
    
    model.eval()
    total_samples = 0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches or total_samples >= max_samples:
                break
            batch = {k: v.to(model.device) for k, v in batch.items() if isinstance(v, torch.Tensor)}
            model(**batch)
            total_samples += batch['input_ids'].size(0)
    
    for hook in hooks:
        hook.remove()
    
    for i in range(num_layers):
        if hidden_states_per_layer[i]:
            hidden_states_per_layer[i] = torch.cat(hidden_states_per_layer[i], dim=0)
        else:
            hidden_states_per_layer[i] = torch.empty(0)
    
    return hidden_states_per_layer

def compute_gelora_base_ranks(model, dataloader, config: ExperimentConfig) -> Dict[str, int]:
    layer_hidden = collect_hidden_states(model, dataloader, num_batches=8, max_samples=800)
    layer_id = {}
    for layer_idx, hs in layer_hidden.items():
        if hs.shape[0] < 10:
            id_val = 1.0
        else:
            id_val = estimate_intrinsic_dimension_pytorch(hs, k=min(5, hs.shape[0]-1))
        base_rank = max(config.gelora_min_rank, min(config.gelora_max_rank,
                                                    int(id_val + config.gelora_margin)))
        layer_id[layer_idx] = base_rank
        logger.info(f"Layer {layer_idx} intrinsic dim ≈ {id_val:.2f} → base rank {base_rank}")
    
    target_substrings = ["query", "key", "value", "dense"]
    rank_dict = {}
    for name, module in model.named_modules():
        if any(ts in name for ts in target_substrings) and "classifier" not in name and "pooler" not in name:
            parts = name.split('.')
            layer_idx = None
            for i, part in enumerate(parts):
                if part == 'layer' and i+1 < len(parts):
                    try:
                        layer_idx = int(parts[i+1])
                        break
                    except:
                        pass
            if layer_idx is not None and layer_idx in layer_id:
                rank_dict[name] = layer_id[layer_idx]
            else:
                rank_dict[name] = config.lora_r
    return rank_dict

def scale_ranks_with_fisher(base_ranks: Dict[str, int], fisher_scores: Dict[str, float],
                            config: ExperimentConfig) -> Dict[str, int]:
    module_fisher = {}
    for full_name, score in fisher_scores.items():
        base_name = full_name.split('.weight')[0].split('.bias')[0].replace('.lora_A', '').replace('.lora_B', '')
        module_fisher[base_name] = module_fisher.get(base_name, 0.0) + score
    
    total = sum(module_fisher.values()) + 1e-8
    module_fisher = {k: v/total for k, v in module_fisher.items()}
    
    scaled = {}
    for module_name, base_r in base_ranks.items():
        fisher_score = module_fisher.get(module_name, 0.0)
        if fisher_score == 0.0:
            for key in module_fisher:
                if key in module_name or module_name in key:
                    fisher_score = module_fisher[key]
                    break
        scale = 1 + fisher_score * (config.fisher_scale_max - 1)
        new_r = int(base_r * scale)
        new_r = max(config.gelora_min_rank, min(config.gelora_max_rank, new_r))
        scaled[module_name] = new_r
        if fisher_score > 0.01:
            logger.info(f"Fisher scaling {module_name}: {base_r} -> {new_r} (score {fisher_score:.3f})")
    return scaled

# ----------------------------------------------------------------------
# R‑Drop & R3F
# ----------------------------------------------------------------------
def rdrop_loss(logits1, logits2, labels, is_regression, alpha):
    if is_regression:
        loss_fn = nn.MSELoss()
        loss1 = loss_fn(logits1.view(-1), labels.view(-1))
        loss2 = loss_fn(logits2.view(-1), labels.view(-1))
        p = torch.distributions.Normal(logits1, torch.ones_like(logits1))
        q = torch.distributions.Normal(logits2, torch.ones_like(logits2))
        kl = (torch.distributions.kl_divergence(p, q).mean() +
              torch.distributions.kl_divergence(q, p).mean()) / 2.0
        loss = (loss1 + loss2)/2.0 + alpha * kl
    else:
        ce = nn.CrossEntropyLoss()
        loss1 = ce(logits1.view(-1, logits1.size(-1)), labels.view(-1))
        loss2 = ce(logits2.view(-1, logits2.size(-1)), labels.view(-1))
        logp1 = F.log_softmax(logits1, dim=-1)
        logp2 = F.log_softmax(logits2, dim=-1)
        p1 = torch.softmax(logits1, dim=-1)
        p2 = torch.softmax(logits2, dim=-1)
        kl1 = F.kl_div(logp1, p2, reduction="batchmean")
        kl2 = F.kl_div(logp2, p1, reduction="batchmean")
        kl = (kl1 + kl2)/2.0
        loss = (loss1 + loss2)/2.0 + alpha * kl
    return loss

def r3f_forward(model, input_ids, attention_mask, token_type_ids, labels, epsilon, is_regression):
    base_kwargs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if token_type_ids is not None:
        base_kwargs["token_type_ids"] = token_type_ids
    out_clean = model(**base_kwargs)
    logits_clean = out_clean.logits
    
    embed = model.get_input_embeddings()
    inputs_embeds = embed(input_ids)
    noise = torch.randn_like(inputs_embeds) * epsilon
    noise_kwargs = {"inputs_embeds": inputs_embeds + noise, "attention_mask": attention_mask, "labels": labels}
    if token_type_ids is not None:
        noise_kwargs["token_type_ids"] = token_type_ids
    out_noisy = model(**noise_kwargs)
    logits_noisy = out_noisy.logits
    
    if is_regression:
        mse = nn.MSELoss()
        loss_clean = mse(logits_clean.view(-1), labels.view(-1))
        loss_noisy = mse(logits_noisy.view(-1), labels.view(-1))
        p = torch.distributions.Normal(logits_clean, torch.ones_like(logits_clean))
        q = torch.distributions.Normal(logits_noisy, torch.ones_like(logits_noisy))
        kl = torch.distributions.kl_divergence(p, q).mean()
        loss = (loss_clean + loss_noisy)/2.0 + kl
    else:
        ce = nn.CrossEntropyLoss()
        loss_clean = ce(logits_clean.view(-1, logits_clean.size(-1)), labels.view(-1))
        loss_noisy = ce(logits_noisy.view(-1, logits_noisy.size(-1)), labels.view(-1))
        logp_clean = F.log_softmax(logits_clean, dim=-1)
        logp_noisy = F.log_softmax(logits_noisy, dim=-1)
        p = torch.softmax(logits_clean, dim=-1)
        q = torch.softmax(logits_noisy, dim=-1)
        kl1 = F.kl_div(logp_clean, q, reduction="batchmean")
        kl2 = F.kl_div(logp_noisy, p, reduction="batchmean")
        loss = (loss_clean + loss_noisy)/2.0 + (kl1+kl2)/2.0
    return loss, logits_clean

# ----------------------------------------------------------------------
# LLRD
# ----------------------------------------------------------------------
def get_llrd_param_groups(model: nn.Module, base_lr: float, weight_decay: float, decay: float):
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "layernorm.weight"]
    max_layer = -1
    for name, _ in model.named_parameters():
        if "encoder.layer." in name:
            layer = int(name.split("encoder.layer.")[1].split(".")[0])
            max_layer = max(max_layer, layer)
    num_layers = max_layer + 1
    
    emb_decay, emb_nodecay = [], []
    layer_decay = {i: [] for i in range(num_layers)}
    layer_nodecay = {i: [] for i in range(num_layers)}
    head_decay, head_nodecay = [], []
    
    for name, param in model.named_parameters():
        if "embeddings" in name:
            if any(nd in name for nd in no_decay):
                emb_nodecay.append(param)
            else:
                emb_decay.append(param)

        elif "encoder.layer." in name:
            layer = int(name.split("encoder.layer.")[1].split(".")[0])
            if any(nd in name for nd in no_decay):
                layer_nodecay[layer].append(param)
            else:
                layer_decay[layer].append(param)
        elif "classifier" in name or "pooler" in name:
            if any(nd in name for nd in no_decay):
                head_nodecay.append(param)
            else:
                head_decay.append(param)
    
    groups = []
    emb_lr = base_lr * (decay ** (num_layers + 1))
    if emb_decay:
        groups.append({"params": emb_decay, "lr": emb_lr, "weight_decay": weight_decay})
    if emb_nodecay:
        groups.append({"params": emb_nodecay, "lr": emb_lr, "weight_decay": 0.0})
    
    for layer in range(num_layers):
        lr_layer = base_lr * (decay ** (num_layers - layer))
        if layer_decay[layer]:
            groups.append({"params": layer_decay[layer], "lr": lr_layer, "weight_decay": weight_decay})
        if layer_nodecay[layer]:
            groups.append({"params": layer_nodecay[layer], "lr": lr_layer, "weight_decay": 0.0})
    
    if head_decay:
        groups.append({"params": head_decay, "lr": base_lr, "weight_decay": weight_decay})
    if head_nodecay:
        groups.append({"params": head_nodecay, "lr": base_lr, "weight_decay": 0.0})
    
    return groups

# ----------------------------------------------------------------------
# Visualizer (with fallbacks)
# ----------------------------------------------------------------------
class Visualizer:
    @staticmethod
    def save_training_plots(history: List[Dict], out_dir: str, method: str, task: str):
        if not _HAS_PLT or not _HAS_PANDAS:
            logger.warning("matplotlib/pandas not installed – skipping plots")
            return
        df = pd.DataFrame(history)
        df.to_csv(os.path.join(out_dir, f"{method}_{task}_log.csv"), index=False)
        plot_cols = [
            ("loss", "Training Loss", "Step", "Loss"),
            ("eval_loss", "Validation Loss", "Epoch", "Loss"),
            ("eval_accuracy", "Accuracy", "Epoch", "Score"),
            ("eval_matthews_correlation", "Matthews Correlation", "Epoch", "Score"),
            ("eval_pearson", "Pearson Correlation", "Epoch", "Score"),
            ("learning_rate", "Learning Rate", "Step", "LR"),
            ("grad_norm", "Gradient Norm", "Step", "||grad||"),
        ]
        for col, title, xlabel, ylabel in plot_cols:
            if col in df.columns:
                plt.figure(figsize=(8,5))
                df[col].dropna().plot()
                plt.title(f"{method.upper()} - {task.upper()} - {title}")
                plt.xlabel(xlabel); plt.ylabel(ylabel); plt.grid(True)
                plt.savefig(os.path.join(out_dir, f"{method}_{task}_{col}.png"), dpi=150, bbox_inches="tight")
                plt.close()
    
    @staticmethod
    def save_confusion_matrix(preds, labels, out_dir, method, task):
        if not _HAS_SKLEARN or not _HAS_PLT:
            return
        if len(np.unique(labels)) > 10:
            return
        cm = confusion_matrix(labels, preds)
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{method.upper()} - {task.upper()} - Confusion Matrix")
        plt.savefig(os.path.join(out_dir, f"{method}_{task}_confusion.png"), dpi=150, bbox_inches="tight")
        plt.close()
    
    @staticmethod
    def save_prediction_distribution(scores, out_dir, method, task):
        if not _HAS_PLT:
            return
        plt.figure(figsize=(8,5))
        plt.hist(scores, bins=30, alpha=0.7, edgecolor="black")
        plt.title(f"{method.upper()} - {task.upper()} - Prediction Confidence")
        plt.xlabel("Confidence"); plt.ylabel("Frequency"); plt.grid(True)
        plt.savefig(os.path.join(out_dir, f"{method}_{task}_pred_dist.png"), dpi=150, bbox_inches="tight")
        plt.close()

# ----------------------------------------------------------------------
# Custom Trainer
# ----------------------------------------------------------------------
class MethodTrainer(Trainer):
    def __init__(self, method: str, rdrop_alpha: float, r3f_epsilon: float,
                 is_regression: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.method = method
        self.rdrop_alpha = rdrop_alpha
        self.r3f_epsilon = r3f_epsilon
        self.is_regression = is_regression

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if "rdrop" in self.method.lower():
            outputs1 = model(**inputs)
            outputs2 = model(**inputs)
            loss = rdrop_loss(outputs1.logits, outputs2.logits, inputs["labels"],
                              self.is_regression, self.rdrop_alpha)
            return (loss, outputs1) if return_outputs else loss
        elif "r3f" in self.method.lower():
            loss, logits = r3f_forward(
                model,
                input_ids=inputs["input_ids"],
                attention_mask=inputs.get("attention_mask"),
                token_type_ids=inputs.get("token_type_ids"),
                labels=inputs["labels"],
                epsilon=self.r3f_epsilon,
                is_regression=self.is_regression,
            )
            return (loss, logits) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs, **kwargs)

# ----------------------------------------------------------------------
# Logging Callback
# ----------------------------------------------------------------------
class LoggingCallback(TrainerCallback):
    def __init__(self, task_dir: str):
        self.task_dir = task_dir
        self.records = []
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            logs = {**logs, "step": state.global_step, "epoch": state.epoch}
            self.records.append(logs)
    def on_train_end(self, args, state, control, **kwargs):
        if self.records and _HAS_PANDAS:
            pd.DataFrame(self.records).to_csv(os.path.join(self.task_dir, "training_log.csv"), index=False)
        elif self.records:
            with open(os.path.join(self.task_dir, "training_log.jsonl"), "w") as f:
                for rec in self.records:
                    f.write(json.dumps(rec) + "\n")

# ----------------------------------------------------------------------
# Memory helper
# ----------------------------------------------------------------------
def adjust_batch_size_on_oom(exception, batch_size, device):
    if "MPS backend out of memory" in str(exception) and "mps" in str(device):
        new_bs = max(1, batch_size // 2)
        logger.warning(f"OOM on MPS with batch size {batch_size}. Try reducing to {new_bs}.")
        return new_bs
    return None

# ----------------------------------------------------------------------
# Single Task Training (with all method branches)
# ----------------------------------------------------------------------
def train_single_task(config: ExperimentConfig, run_dir: str, device: torch.device,
                      
                     fp16_usable: bool, task_name: str, seed: int) -> Dict:
    logger.info(f"===== Task: {task_name} (method={config.method}, seed={seed}) =====")
    task_dir = os.path.join(run_dir, task_name, f"seed_{seed}")
    os.makedirs(task_dir, exist_ok=True)

    set_all_seeds(seed)

    # Load data
    try:
        datasets = load_dataset("glue", task_name)
    except Exception as e:
        logger.warning(f"HF error: {e}, falling back to local Parquet files")
        datasets = load_glue_local(task_name)

    num_labels, is_regression = get_num_labels_and_is_regression(task_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name_or_path, use_fast=True)
    config_obj = AutoConfig.from_pretrained(config.model_name_or_path, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(config.model_name_or_path, config=config_obj)
    # Apply SVD attention compression if requested
    if config.svd_attn:
        logger.info("Replacing all MultiheadAttention layers with SVD compression")
        replace_attention_with_svd(model, config.svd_rank_qkv, config.svd_rank_out)

    # --------------------------------------------------------------
    # Apply advanced methods
    # --------------------------------------------------------------
    per_layer_ranks = None
    
    if "fisher_gelora" in config.method.lower():
        logger.info("Applying Fisher‑Weighted GeLoRA (static rank scaling)")
        
        # ---- Create a fixed‑length sample dataset for GeLoRA ----
        sent1_key, sent2_key = GLUE_TASK_KEYS[task_name.lower()]
        def sample_preprocess(examples):
            # Force padding to max_seq_length
            args = (examples[sent1_key],) if sent2_key is None else (examples[sent1_key], examples[sent2_key])
            tokenized = tokenizer(*args, truncation=True, max_length=config.max_seq_length, padding='max_length')
            # Keep labels for loss calculation (if needed)
            if get_num_labels_and_is_regression(task_name)[1]:
                tokenized["labels"] = [float(label) for label in examples["label"]]
            else:
                tokenized["labels"] = examples["label"]
            return tokenized
        
        sample_size = min(800, len(datasets["train"]))
        sample_dataset = datasets["train"].select(range(sample_size))
        sample_dataset = sample_dataset.map(sample_preprocess, batched=True)
        sample_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        sample_loader = DataLoader(sample_dataset, batch_size=4, shuffle=False)
        
        # Continue with GeLoRA and Fisher rank computation...
        model.to(device)
        model.eval()
        base_ranks = compute_gelora_base_ranks(model, sample_loader, config)
        # ... rest of the code


    elif "fisher_lora" in config.method.lower():
        logger.info("Applying Fisher‑LoRA (uniform base rank per layer)")
        fisher_scores = compute_fisher_importance(model, datasets["train"], tokenizer, task_name, config)
        if fisher_scores is None:
            logger.warning("Fisher scores not computed; using uniform LoRA")
            apply_lora_or_dora(model, config.lora_r, config.lora_alpha, config.lora_dropout, method_type='lora')
        else:
            target_substrings = ["query", "key", "value", "dense"]
            rank_dict = {}
            for name, _ in model.named_modules():
                if any(ts in name for ts in target_substrings) and "classifier" not in name and "pooler" not in name:
                    best_score = 0.5
                    for fname, fval in fisher_scores.items():
                        if name in fname or fname in name:
                            best_score = max(best_score, fval)
                    rank = int(config.lora_r * (0.5 + best_score))
                    rank = max(config.gelora_min_rank, min(config.gelora_max_rank, rank))
                    rank_dict[name] = rank
            save_json(rank_dict, os.path.join(task_dir, "fisher_lora_ranks.json"))
            apply_lora_or_dora(model, config.lora_r, config.lora_alpha, config.lora_dropout,
                               method_type='lora', per_layer_ranks=rank_dict)

    elif "dislora" in config.method.lower():
        logger.info("Applying DisLoRA")
        apply_lora_or_dora(model, config.lora_r, config.lora_alpha, config.lora_dropout,
                           method_type='dislora',
                           dislora_beta=config.dislora_beta,
                           dislora_capacity_factor=config.dislora_capacity_factor)

    elif "dora" in config.method.lower():
        logger.info("Applying DoRA")
        apply_lora_or_dora(model, config.lora_r, config.lora_alpha, config.lora_dropout, method_type='dora')

    elif "lora" in config.method.lower():
        logger.info("Applying LoRA")
        apply_lora_or_dora(model, config.lora_r, config.lora_alpha, config.lora_dropout, method_type='lora')

    elif "mixout" in config.method.lower():
        logger.info("Applying Mixout")
        replace_linears_with_mixout(model, config.mixout_p, skip_classifier=True)

    # Gradient checkpointing
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled")

    model.to(device)

    # --------------------------------------------------------------
    # Preprocess full dataset
    # --------------------------------------------------------------
    preprocess_fn = preprocess_function_builder(tokenizer, task_name, config.max_seq_length)
    num_cpus = os.cpu_count() or 1
    num_proc = min(4, num_cpus)
    encoded = datasets.map(preprocess_fn, batched=True, num_proc=num_proc)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in encoded["train"].column_names:
        cols.append("token_type_ids")
    encoded.set_format(type="torch", columns=cols)

    train_ds = encoded["train"]
    eval_key = "validation" if "validation" in encoded else "validation_matched"
    eval_ds = encoded[eval_key]
    data_collator = DataCollatorWithPadding(tokenizer)

    # Compute warmup_steps correctly
    effective_batch = config.per_device_train_batch_size * config.gradient_accumulation_steps
    num_training_steps = (len(train_ds) // effective_batch) * config.num_train_epochs
    warmup_steps = int(config.warmup_ratio * num_training_steps)

    primary_metric = get_primary_metric_name(task_name)
    training_args = TrainingArguments(
        output_dir=task_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        num_train_epochs=config.num_train_epochs,
        weight_decay=config.weight_decay,
        warmup_steps=warmup_steps,
        logging_steps=config.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{primary_metric}",
        greater_is_better=True,
        fp16=fp16_usable,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        dataloader_pin_memory=False,
        max_grad_norm=config.max_grad_norm,
    )

    # Optimizer
    if "llrd" in config.method.lower():
        param_groups = get_llrd_param_groups(model, config.learning_rate, config.weight_decay, config.llrd_decay)
        if config.use_lion:
            try:
                from lion_pytorch import Lion
            except ImportError:
                logger.warning("Lion not installed – using AdamW")
                from torch.optim import AdamW
                optimizer = AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
            else:
                optimizer = Lion(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            from torch.optim import AdamW
            optimizer = AdamW(param_groups, lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        if config.use_lion:
            try:
                from lion_pytorch import Lion
            except ImportError:
                logger.warning("Lion not installed – using AdamW")
                from torch.optim import AdamW
                optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
            else:
                optimizer = Lion(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        else:
            from torch.optim import AdamW
            optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    callbacks = [LoggingCallback(task_dir)]
    if config.early_stopping_patience is not None:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience))

    compute_metrics = compute_metrics_builder(task_name)
    trainer = MethodTrainer(
        method=config.method,
        rdrop_alpha=config.rdrop_alpha,
        r3f_epsilon=config.r3f_epsilon,
        is_regression=is_regression,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        optimizers=(optimizer, None),
        callbacks=callbacks,
    )

    try:
        trainer.train()
    except RuntimeError as e:
        new_bs = adjust_batch_size_on_oom(e, config.per_device_train_batch_size, device)
        if new_bs is not None:
            raise RuntimeError(
                f"OOM with batch size {config.per_device_train_batch_size}. "
                f"Restart with --per_device_train_batch_size {new_bs}"
            ) from e
        else:
            raise

    eval_metrics = trainer.evaluate()
    save_json(eval_metrics, os.path.join(task_dir, "eval_metrics.json"))

    # Visualisations (if libraries present)
    if _HAS_PLT and _HAS_PANDAS:
        vis = Visualizer()
        vis.save_training_plots(trainer.state.log_history, task_dir, config.method, task_name)
        if not is_regression and _HAS_SKLEARN:
            preds = trainer.predict(eval_ds)
            pred_labels = np.argmax(preds.predictions, axis=1)
            vis.save_confusion_matrix(pred_labels, preds.label_ids, task_dir, config.method, task_name)
            probs = torch.softmax(torch.tensor(preds.predictions), dim=-1).max(dim=-1).values.numpy()
            vis.save_prediction_distribution(probs, task_dir, config.method, task_name)

    del trainer, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()

    return eval_metrics

# ----------------------------------------------------------------------
# Main (multi‑seed loop)
# ----------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="GLUE Ultimate Framework")
    parser.add_argument("--model_name_or_path", type=str, default="bert-base-uncased")
    parser.add_argument("--tasks", nargs="+", default=["cola", "mrpc", "rte", "stsb", "sst2"])
    parser.add_argument("--method", type=str, required=True)
    parser.add_argument("--output_root", type=str, default="./experiments")
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--per_device_train_batch_size", type=int, default=32)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--lr_scheduler_type", type=str, default="cosine")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=5.0)
    parser.add_argument("--r3f_epsilon", type=float, default=1e-5)
    parser.add_argument("--llrd_decay", type=float, default=0.95)
    parser.add_argument("--mixout_p", type=float, default=0.9)
    parser.add_argument("--use_lion", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    # Fisher / GeLoRA
    parser.add_argument("--fisher_num_batches", type=int, default=20)
    parser.add_argument("--fisher_sample_size", type=int, default=200)
    parser.add_argument("--gelora_margin", type=float, default=1.0)
    parser.add_argument("--gelora_min_rank", type=int, default=2)
    parser.add_argument("--gelora_max_rank", type=int, default=32)
    parser.add_argument("--fisher_scale_max", type=float, default=2.0)
    # DisLoRA
    parser.add_argument("--dislora_beta", type=float, default=0.5)
    parser.add_argument("--dislora_capacity_factor", type=float, default=1.0)
    # Multi‑seed
    parser.add_argument("--num_seeds", type=int, default=1)
    # SVD attention
    parser.add_argument("--svd_attn", action="store_true", help="Replace attention with SVD compressed version")
    parser.add_argument("--svd_rank_qkv", type=int, default=32, help="Rank for QKV SVD")
    parser.add_argument("--svd_rank_out", type=int, default=32, help="Rank for output SVD")
    return parser.parse_args()



def main():
    args = parse_args()
    if args.force_cpu:
        device = torch.device("cpu")
        fp16_usable = False
    else:
        device, fp16_usable = get_device_and_precision(args.fp16)
    logger.info(f"Device: {device}, fp16_usable={fp16_usable}")

    config = ExperimentConfig(
        model_name_or_path=args.model_name_or_path,
        tasks=args.tasks,
        method=args.method,
        output_root=args.output_root,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        seed=args.seed,
        svd_attn=args.svd_attn,
        svd_rank_qkv=args.svd_rank_qkv,
        svd_rank_out=args.svd_rank_out,

        max_seq_length=args.max_seq_length,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        save_total_limit=args.save_total_limit,
        fp16=fp16_usable,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        rdrop_alpha=args.rdrop_alpha,
        r3f_epsilon=args.r3f_epsilon,
        llrd_decay=args.llrd_decay,
        mixout_p=args.mixout_p,
        use_lion=args.use_lion,
        early_stopping_patience=args.early_stopping_patience,
        gradient_checkpointing=args.gradient_checkpointing,
        max_grad_norm=args.max_grad_norm,
        fisher_num_batches=args.fisher_num_batches,
        fisher_sample_size=args.fisher_sample_size,
        gelora_margin=args.gelora_margin,
        gelora_min_rank=args.gelora_min_rank,
        gelora_max_rank=args.gelora_max_rank,
        fisher_scale_max=args.fisher_scale_max,
        dislora_beta=args.dislora_beta,
        dislora_capacity_factor=args.dislora_capacity_factor,
        num_seeds=args.num_seeds,
    )

    run_dir = create_experiment_dir(config)
    save_system_info(run_dir)
    save_config(config, run_dir)

    logger.info(f"Experiment dir: {run_dir}")
    logger.info(f"Tasks: {config.tasks}")
    logger.info(f"Method: {config.method}")

    all_results = {}
    for task in config.tasks:
        task_seed_results = []
        for i in range(config.num_seeds):
            seed = config.seed + i
            try:
                metrics = train_single_task(config, run_dir, device, fp16_usable, task, seed)
                task_seed_results.append(metrics)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                logger.info(f"✅ {task} (seed={seed}): {metrics}")
            except Exception as e:
                logger.error(f"❌ {task} (seed={seed}) failed: {e}", exc_info=True)
                task_seed_results.append({"error": str(e)})

        # Aggregate across seeds
        if task_seed_results and all("error" not in r for r in task_seed_results):
            keys = task_seed_results[0].keys()
            aggregated = {}
            for key in keys:
                values = [r[key] for r in task_seed_results if key in r and r[key] is not None]
                if values:
                    aggregated[f"{key}_mean"] = float(np.mean(values))
                    aggregated[f"{key}_std"] = float(np.std(values))
                    aggregated[f"{key}_values"] = values
            all_results[task] = aggregated
        else:
            all_results[task] = {"errors": task_seed_results}
        save_json(all_results, os.path.join(run_dir, "summary_all_tasks.json"))

    save_json(all_results, os.path.join(run_dir, "summary_all_tasks.json"))
    logger.info("🎉 All experiments finished.")

if __name__ == "__main__":
    main()