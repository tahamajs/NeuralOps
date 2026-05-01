#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
GLUE Fine-tuning with SVD + LoRA Compression
Complete production version with all features
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
    get_linear_schedule_with_warmup,
)
from datasets import Dataset, DatasetDict
import evaluate
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from tqdm import tqdm

# ---------- Offline settings ----------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# ---------- GLUE task configuration ----------
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
        split = fname.split("-")[0]  # e.g., "train", "validation"
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

def get_num_labels_and_is_regression(task_name: str):
    """Get number of labels and regression flag for GLUE task."""
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

def get_primary_metric(task_name: str):
    """Get primary evaluation metric for GLUE task."""
    t = task_name.lower()
    if t == "cola":
        return "matthews_correlation", True
    if t == "stsb":
        return "pearson_spearman", True
    if t in ["mrpc", "qqp"]:
        return "f1", True
    return "accuracy", True

def compute_metrics_builder(task_name):
    """Build metrics computation function for GLUE task."""
    try:
        metric = evaluate.load("glue", task_name)
    except:
        metric = evaluate.load("accuracy")
    
    _, is_reg = get_num_labels_and_is_regression(task_name)

    def compute_metrics(p):
        preds = p.predictions
        labels = p.label_ids
        if not is_reg:
            preds = np.argmax(preds, axis=1)
        else:
            preds = preds.squeeze()
        result = metric.compute(predictions=preds, references=labels)
        
        # Special handling for STS-B
        if task_name == "stsb" and "pearson" in result and "spearmanr" in result:
            result["pearson_spearman"] = (result["pearson"] + result["spearmanr"]) / 2.0
        
        return result
    return compute_metrics

# ----------------------------------------------------------------------
# Compression Modules
# ----------------------------------------------------------------------
class SVDLinear(nn.Module):
    """
    Low-rank linear layer: W = U @ V
    U: (out_features, rank)   V: (rank, in_features)
    Forward: x -> (U @ V) x + bias   computed efficiently as U(Vx) + bias
    """
    def __init__(self, in_features: int, out_features: int, rank: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        
        # Low-rank factors
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        
        # Optional bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters with orthogonal matrices for better gradient flow."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> Vx -> U(Vx) + bias."""
        # Efficient computation: (U @ V) x = U (V x)
        Vx = F.linear(x, self.V)      # shape: (..., rank)
        Ux = F.linear(Vx, self.U)     # shape: (..., out_features)
        
        if self.bias is not None:
            Ux = Ux + self.bias
        
        return Ux
    
    def set_low_rank_factors(self, U: torch.Tensor, V: torch.Tensor):
        """Initialize from pre-computed SVD factors."""
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}'

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer: W = W_orig + A @ B
    A: (out_features, rank)   B: (rank, in_features)
    Forward: x -> (W_orig + A @ B) x + bias
    """
    def __init__(self, original_layer: nn.Linear, rank: int, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.original_layer = original_layer
        self.rank = rank
        self.alpha = alpha
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Freeze original weights
        for param in self.original_layer.parameters():
            param.requires_grad = False
        
        # LoRA adapters
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize LoRA parameters."""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> (W_orig + (alpha/rank) * B @ A) x + bias."""
        # Original forward
        orig_output = self.original_layer(x)
        
        # LoRA forward
        lora_output = F.linear(self.dropout(x), self.lora_A)  # shape: (..., rank)
        lora_output = F.linear(lora_output, self.lora_B)      # shape: (..., out_features)
        
        # Scale by alpha/rank
        lora_output = lora_output * (self.alpha / self.rank)
        
        return orig_output + lora_output
    
    def extra_repr(self) -> str:
        return f'rank={self.rank}, alpha={self.alpha}, dropout={self.dropout.p if hasattr(self.dropout, "p") else 0.0}'

class AdaptiveSVDLinear(nn.Module):
    """
    Adaptive SVD layer with residual factorization for better approximation.
    W = U @ V + A @ B where U,V from SVD and A,B are trainable residuals.
    """
    def __init__(self, in_features: int, out_features: int, rank: int, 
                 residual_rank: int = 0, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.residual_rank = residual_rank
        
        # SVD factors
        self.U = nn.Parameter(torch.empty(out_features, rank))
        self.V = nn.Parameter(torch.empty(rank, in_features))
        
        # Residual factors (if enabled)
        if residual_rank > 0:
            self.A = nn.Parameter(torch.empty(out_features, residual_rank))
            self.B = nn.Parameter(torch.empty(residual_rank, in_features))
        else:
            self.register_parameter('A', None)
            self.register_parameter('B', None)
        
        # Bias
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        
        if self.A is not None and self.B is not None:
            nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: x -> (U@V + A@B) x + bias."""
        # SVD part: U(Vx)
        Vx = F.linear(x, self.V)      # shape: (..., rank)
        svd_output = F.linear(Vx, self.U)  # shape: (..., out_features)
        
        # Residual part: A(Bx) if enabled
        if self.A is not None and self.B is not None:
            Bx = F.linear(x, self.B)      # shape: (..., residual_rank)
            residual_output = F.linear(Bx, self.A)  # shape: (..., out_features)
            output = svd_output + residual_output
        else:
            output = svd_output
        
        # Add bias
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def set_svd_factors(self, U: torch.Tensor, V: torch.Tensor):
        """Initialize SVD factors from pre-computed decomposition."""
        with torch.no_grad():
            self.U.copy_(U)
            self.V.copy_(V)
    
    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, residual_rank={self.residual_rank}, bias={self.bias is not None}'

# ----------------------------------------------------------------------
# Compression Strategies
# ----------------------------------------------------------------------
def compress_with_svd(
    model: nn.Module,
    rank: int = 64,
    target_names: Optional[List[str]] = None,
    skip_classifier: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Replace Linear layers with SVDLinear.
    
    Args:
        model: The model to compress
        rank: Fixed rank for SVD compression
        target_names: List of substring patterns to match layer names
        skip_classifier: Whether to skip classifier layers
        verbose: Print compression details
    
    Returns:
        Compressed model
    """
    if target_names is None:
        target_names = [
            "query", "key", "value", "dense",
            "intermediate", "output", "pooler",
        ]
    
    compressed_layers = []
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # Skip classifier layers if requested
        if skip_classifier and "classifier" in name.lower():
            if verbose:
                print(f"  Skipping (classifier): {name}")
            continue
        
        # Check if layer name contains any target pattern
        if not any(pattern in name.lower() for pattern in [p.lower() for p in target_names]):
            continue
        
        # Get original weight
        W_orig = module.weight.data.cpu().numpy()
        out_dim, in_dim = W_orig.shape
        
        # Determine effective rank
        effective_rank = min(rank, out_dim, in_dim)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(W_orig, full_matrices=False)
        
        # Keep top singular values/vectors
        U_r = torch.tensor(U[:, :effective_rank], dtype=module.weight.dtype)
        V_r = torch.tensor(
            np.diag(S[:effective_rank]) @ Vt[:effective_rank, :],
            dtype=module.weight.dtype,
        )
        
        # Create SVD layer
        svd_linear = SVDLinear(
            in_dim, out_dim, effective_rank,
            bias=module.bias is not None
        )
        svd_linear.set_low_rank_factors(U_r, V_r)
        
        # Replace in model
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], svd_linear)
        
        compressed_layers.append((name, effective_rank, in_dim * out_dim, 
                                 effective_rank * (in_dim + out_dim)))
        
        if verbose:
            print(f"  Compressed {name}: {in_dim}x{out_dim} -> rank {effective_rank} "
                  f"({effective_rank * (in_dim + out_dim):,} params vs {in_dim * out_dim:,} original)")
    
    # Print summary
    if verbose and compressed_layers:
        print("\n" + "="*60)
        print("SVD Compression Summary:")
        print("="*60)
        total_orig = sum(l[2] for l in compressed_layers)
        total_comp = sum(l[3] for l in compressed_layers)
        for name, rank, orig, comp in compressed_layers:
            reduction = 1 - comp/orig
            print(f"  {name:40s} rank={rank:3d} | {comp:8,} / {orig:8,} params "
                  f"({comp/orig:.1%}, -{reduction:.1%})")
        print(f"\n  Total: {total_comp:,} / {total_orig:,} params "
              f"({total_comp/total_orig:.1%}, -{1-total_comp/total_orig:.1%})")
        print("="*60)
    
    return model

def compress_with_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
    target_names: Optional[List[str]] = None,
    skip_classifier: bool = False,
    verbose: bool = True
) -> nn.Module:
    """
    Apply LoRA to Linear layers.
    
    Args:
        model: The model to compress
        rank: LoRA rank
        alpha: LoRA scaling factor
        dropout: Dropout rate for LoRA
        target_names: List of substring patterns to match layer names
        skip_classifier: Whether to skip classifier layers
        verbose: Print compression details
    
    Returns:
        Model with LoRA adapters
    """
    if target_names is None:
        target_names = [
            "query", "key", "value", "dense",
            "intermediate", "output", "pooler",
        ]
    
    lora_layers = []
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # Skip classifier layers if requested
        if skip_classifier and "classifier" in name.lower():
            if verbose:
                print(f"  Skipping (classifier): {name}")
            continue
        
        # Check if layer name contains any target pattern
        if not any(pattern in name.lower() for pattern in [p.lower() for p in target_names]):
            continue
        
        # Create LoRA layer
        lora_linear = LoRALinear(module, rank, alpha, dropout)
        
        # Replace in model
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], lora_linear)
        
        orig_params = module.in_features * module.out_features
        if module.bias is not None:
            orig_params += module.out_features
        lora_params = rank * (module.in_features + module.out_features)
        
        lora_layers.append((name, rank, orig_params, lora_params))
        
        if verbose:
            print(f"  LoRA applied to {name}: rank={rank}, alpha={alpha}")
    
    # Print summary
    if verbose and lora_layers:
        print("\n" + "="*60)
        print("LoRA Compression Summary:")
        print("="*60)
        total_orig = sum(l[2] for l in lora_layers)
        total_lora = sum(l[3] for l in lora_layers)
        for name, rank, orig, lora in lora_layers:
            print(f"  {name:40s} rank={rank:3d} | {lora:8,} LoRA params")
        print(f"\n  Total LoRA params: {total_lora:,} "
              f"(added to {total_orig:,} frozen params)")
        print("="*60)
    
    return model

def compress_with_adaptive_svd(
    model: nn.Module,
    svd_rank: int = 64,
    residual_rank: int = 8,
    target_names: Optional[List[str]] = None,
    skip_classifier: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Apply Adaptive SVD compression (SVD + trainable residuals).
    
    Args:
        model: The model to compress
        svd_rank: Rank for SVD part
        residual_rank: Rank for residual factorization
        target_names: List of substring patterns to match layer names
        skip_classifier: Whether to skip classifier layers
        verbose: Print compression details
    
    Returns:
        Compressed model
    """
    if target_names is None:
        target_names = [
            "query", "key", "value", "dense",
            "intermediate", "output", "pooler",
        ]
    
    compressed_layers = []
    
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        
        # Skip classifier layers if requested
        if skip_classifier and "classifier" in name.lower():
            if verbose:
                print(f"  Skipping (classifier): {name}")
            continue
        
        # Check if layer name contains any target pattern
        if not any(pattern in name.lower() for pattern in [p.lower() for p in target_names]):
            continue
        
        # Get original weight
        W_orig = module.weight.data.cpu().numpy()
        out_dim, in_dim = W_orig.shape
        
        # Determine effective SVD rank
        effective_svd_rank = min(svd_rank, out_dim, in_dim)
        
        # Perform SVD
        U, S, Vt = np.linalg.svd(W_orig, full_matrices=False)
        
        # Keep top singular values/vectors
        U_r = torch.tensor(U[:, :effective_svd_rank], dtype=module.weight.dtype)
        V_r = torch.tensor(
            np.diag(S[:effective_svd_rank]) @ Vt[:effective_svd_rank, :],
            dtype=module.weight.dtype,
        )
        
        # Create Adaptive SVD layer
        adaptive_svd = AdaptiveSVDLinear(
            in_dim, out_dim, effective_svd_rank, residual_rank,
            bias=module.bias is not None
        )
        adaptive_svd.set_svd_factors(U_r, V_r)
        
        # Replace in model
        parent = model.get_submodule(".".join(name.split(".")[:-1]))
        setattr(parent, name.split(".")[-1], adaptive_svd)
        
        # Calculate parameter counts
        orig_params = in_dim * out_dim
        if module.bias is not None:
            orig_params += out_dim
        
        svd_params = effective_svd_rank * (in_dim + out_dim)
        residual_params = residual_rank * (in_dim + out_dim) if residual_rank > 0 else 0
        total_params = svd_params + residual_params
        if module.bias is not None:
            total_params += out_dim
        
        compressed_layers.append((name, effective_svd_rank, residual_rank, 
                                 orig_params, total_params))
        
        if verbose:
            print(f"  Adaptive SVD applied to {name}: "
                  f"SVD rank={effective_svd_rank}, residual rank={residual_rank}")
    
    # Print summary
    if verbose and compressed_layers:
        print("\n" + "="*60)
        print("Adaptive SVD Compression Summary:")
        print("="*60)
        total_orig = sum(l[3] for l in compressed_layers)
        total_comp = sum(l[4] for l in compressed_layers)
        for name, svd_r, res_r, orig, comp in compressed_layers:
            reduction = 1 - comp/orig
            print(f"  {name:40s} SVD={svd_r:3d}+{res_r:3d} | "
                  f"{comp:8,} / {orig:8,} params ({comp/orig:.1%}, -{reduction:.1%})")
        print(f"\n  Total: {total_comp:,} / {total_orig:,} params "
              f"({total_comp/total_orig:.1%}, -{1-total_comp/total_orig:.1%})")
        print("="*60)
    
    return model

def compress_with_hybrid(
    model: nn.Module,
    svd_rank: int = 64,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.0,
    target_names_svd: Optional[List[str]] = None,
    target_names_lora: Optional[List[str]] = None,
    skip_classifier: bool = True,
    verbose: bool = True
) -> nn.Module:
    """
    Apply hybrid compression: SVD for large layers, LoRA for attention layers.
    
    Args:
        model: The model to compress
        svd_rank: Rank for SVD compression
        lora_rank: Rank for LoRA adapters
        lora_alpha: LoRA scaling factor
        lora_dropout: Dropout rate for LoRA
        target_names_svd: Layers to compress with SVD
        target_names_lora: Layers to apply LoRA to
        skip_classifier: Whether to skip classifier layers
        verbose: Print compression details
    
    Returns:
        Compressed model
    """
    if target_names_svd is None:
        target_names_svd = ["dense", "intermediate", "output", "pooler"]
    
    if target_names_lora is None:
        target_names_lora = ["query", "key", "value"]
    
    # First apply SVD to specified layers
    model = compress_with_svd(
        model, svd_rank, target_names_svd, 
        skip_classifier, verbose=False
    )
    
    # Then apply LoRA to other layers
    model = compress_with_lora(
        model, lora_rank, lora_alpha, lora_dropout,
        target_names_lora, skip_classifier, verbose=False
    )
    
    if verbose:
        print("\n" + "="*60)
        print("Hybrid Compression Applied:")
        print("="*60)
        print(f"  SVD compression (rank={svd_rank}) applied to: {target_names_svd}")
        print(f"  LoRA adapters (rank={lora_rank}, alpha={lora_alpha}) applied to: {target_names_lora}")
        print("="*60)
    
    return model

# ----------------------------------------------------------------------
# Custom Trainer with Gradient Checkpointing
# ----------------------------------------------------------------------
class EfficientTrainer(Trainer):
    """Custom trainer with gradient checkpointing support."""
    
    def __init__(self, *args, use_gradient_checkpointing: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if use_gradient_checkpointing and self.model.supports_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            print("Gradient checkpointing enabled")
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Override to handle custom loss computation if needed.
        """
        outputs = model(**inputs)
        loss = outputs.loss
        
        # Apply gradient accumulation manually if needed
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        
        return (loss, outputs) if return_outputs else loss

# ----------------------------------------------------------------------
# Main Training Function
# ----------------------------------------------------------------------
def fine_tune(
    model: nn.Module,
    tokenizer: AutoTokenizer,
    task_name: str,
    output_dir: str,
    args: argparse.Namespace
) -> nn.Module:
    """
    Fine-tune the compressed model on GLUE task.
    
    Args:
        model: The model to fine-tune
        tokenizer: Tokenizer for the model
        task_name: GLUE task name
        output_dir: Directory to save results
        args: Command line arguments
    
    Returns:
        Fine-tuned model
    """
    # Load dataset
    datasets = load_glue_local(task_name)
    num_labels, is_regression = get_num_labels_and_is_regression(task_name)
    sent1_key, sent2_key = GLUE_TASK_KEYS[task_name]
    
    # Preprocessing function
    def preprocess(examples):
        if sent2_key is None:
            tokenized = tokenizer(
                examples[sent1_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        else:
            tokenized = tokenizer(
                examples[sent1_key],
                examples[sent2_key],
                truncation=True,
                padding="max_length",
                max_length=args.max_seq_length,
            )
        
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        
        return tokenized
    
    # Preprocess datasets
    dataset = datasets.map(preprocess, batched=True)
    train_ds = dataset["train"]
    
    # Determine evaluation dataset
    if task_name == "mnli":
        eval_key = "validation_matched"
    else:
        eval_key = "validation"
    eval_ds = dataset[eval_key]
    
    # Get metrics
    metric_name, greater_is_better = get_primary_metric(task_name)
    compute_metrics = compute_metrics_builder(task_name)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        # overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{metric_name}",
        greater_is_better=greater_is_better,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16 and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        dataloader_drop_last=True,
        label_names=["labels"],
        report_to="none",  # Disable wandb/tensorboard by default
        ddp_find_unused_parameters=False if args.ddp else None,
    )
    
    # Create trainer
    trainer_class = EfficientTrainer if args.use_custom_trainer else Trainer
    
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        # tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train
    print(f"\nStarting fine-tuning on {task_name}...")
    train_result = trainer.train()
    
    # Evaluate
    eval_result = trainer.evaluate()
    
    # Save results
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Fine-tuning results for {task_name}:")
    print(f"{'='*60}")
    for k, v in eval_result.items():
        if isinstance(v, (int, float)):
            print(f"  {k}: {v:.4f}")
    
    # Save training metrics
    metrics_file = os.path.join(output_dir, "training_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Task: {task_name}\n")
        f.write(f"Compression: {args.compression_method}\n")
        f.write(f"Rank: {args.rank}\n")
        f.write(f"Epochs: {args.epochs}\n")
        f.write(f"Learning rate: {args.learning_rate}\n")
        f.write(f"Batch size: {args.batch_size}\n")
        f.write("\nEvaluation Results:\n")
        for k, v in eval_result.items():
            if isinstance(v, (int, float)):
                f.write(f"{k}: {v:.4f}\n")
    
    return model

# ----------------------------------------------------------------------
# Parameter Counting Utilities
# ----------------------------------------------------------------------
def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """Count parameters in model."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())

def print_parameter_summary(model: nn.Module, original_params: int):
    """Print detailed parameter summary."""
    total_params = count_parameters(model)
    trainable_params = count_parameters(model, trainable_only=True)
    
    print(f"\n{'='*60}")
    print("Parameter Summary:")
    print(f"{'='*60}")
    print(f"Original parameters:      {original_params:,}")
    print(f"Total parameters:         {total_params:,}")
    print(f"Trainable parameters:     {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print(f"\nCompression ratio:        {total_params/original_params:.2%}")
    print(f"Trainable ratio:         {trainable_params/total_params:.2%}")
    print(f"Memory reduction:        {1 - total_params/original_params:.2%}")
    print(f"{'='*60}")
    
    # Count by module type
    module_counts = {}
    for name, module in model.named_modules():
        if isinstance(module, (SVDLinear, LoRALinear, AdaptiveSVDLinear, nn.Linear)):
            module_type = module.__class__.__name__
            params = sum(p.numel() for p in module.parameters())
            module_counts[module_type] = module_counts.get(module_type, 0) + params
    
    if module_counts:
        print("\nParameters by module type:")
        for module_type, params in sorted(module_counts.items()):
            print(f"  {module_type:20s}: {params:>12,} params")

# ----------------------------------------------------------------------
# Main Function
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="GLUE fine-tuning with SVD + LoRA compression"
    )
    
    # Task and model
    parser.add_argument("--task", type=str, required=True,
                       choices=["cola", "sst2", "mrpc", "rte", "stsb",
                                "mnli", "qnli", "qqp", "wnli"],
                       help="GLUE task name")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased",
                       help="Base model name")
    
    # Compression settings
    parser.add_argument("--compression_method", type=str, default="svd",
                       choices=["svd", "lora", "adaptive_svd", "hybrid", "none"],
                       help="Compression method")
    parser.add_argument("--rank", type=int, default=64,
                       help="Rank for SVD/LoRA compression")
    parser.add_argument("--residual_rank", type=int, default=8,
                       help="Residual rank for adaptive SVD")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                       help="LoRA dropout rate")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01,
                       help="Weight decay")
    parser.add_argument("--warmup_ratio", type=float, default=0.06,
                       help="Warmup ratio")
    parser.add_argument("--max_seq_length", type=int, default=128,
                       help="Maximum sequence length")
    
    # Advanced settings
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                       help="Gradient accumulation steps")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                       help="Enable gradient checkpointing")
    parser.add_argument("--use_custom_trainer", action="store_true",
                       help="Use custom EfficientTrainer")
    parser.add_argument("--fp16", action="store_true",
                       help="Use mixed precision (FP16)")
    parser.add_argument("--bf16", action="store_true",
                       help="Use bfloat16 precision")
    parser.add_argument("--ddp", action="store_true",
                       help="Use distributed data parallel")
    
    # Other
    parser.add_argument("--logging_steps", type=int, default=10,
                       help="Logging steps")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--output_dir", type=str, default="./compressed_model",
                       help="Output directory")
    parser.add_argument("--skip_classifier", action="store_true",
                       help="Skip compressing classifier layers")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device setup
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model: {args.model_name}")
    num_labels, _ = get_num_labels_and_is_regression(args.task)
    
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=num_labels,
        local_files_only=True,
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        config=config,
        local_files_only=True,
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        local_files_only=True,
    )
    
    # Count original parameters
    original_params = count_parameters(model)
    print(f"Original parameters: {original_params:,}")
    
    # Apply compression
    if args.compression_method != "none":
        print(f"\nApplying {args.compression_method} compression...")
        
        if args.compression_method == "svd":
            model = compress_with_svd(
                model,
                rank=args.rank,
                skip_classifier=args.skip_classifier,
                verbose=True
            )
        
        elif args.compression_method == "lora":
            model = compress_with_lora(
                model,
                rank=args.rank,
                alpha=args.lora_alpha,
                dropout=args.lora_dropout,
                skip_classifier=args.skip_classifier,
                verbose=True
            )
        
        elif args.compression_method == "adaptive_svd":
            model = compress_with_adaptive_svd(
                model,
                svd_rank=args.rank,
                residual_rank=args.residual_rank,
                skip_classifier=args.skip_classifier,
                verbose=True
            )
        
        elif args.compression_method == "hybrid":
            model = compress_with_hybrid(
                model,
                svd_rank=args.rank,
                lora_rank=args.rank // 8,  # Default to 1/8 of SVD rank
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                skip_classifier=args.skip_classifier,
                verbose=True
            )
    
    # Print parameter summary
    print_parameter_summary(model, original_params)
    
    # Fine-tune
    model = fine_tune(model, tokenizer, args.task, args.output_dir, args)
    
    print(f"\n✅ Training complete! Model saved to: {args.output_dir}")
    
    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Model Statistics:")
    print(f"{'='*60}")
    final_params = count_parameters(model)
    final_trainable = count_parameters(model, trainable_only=True)
    print(f"Total parameters:     {final_params:,}")
    print(f"Trainable parameters: {final_trainable:,}")
    print(f"Compression ratio:    {final_params/original_params:.2%}")
    print(f"Memory saved:         {1 - final_params/original_params:.2%}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
