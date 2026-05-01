#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Token Compression Attention for GLUE — Production-Ready Version
================================================================
Fixes all critical bugs and adds:
1. Proper weight initialization from pretrained BERT
2. Correct value compression with orthogonal projection
3. Gradient checkpointing for memory efficiency
4. LayerNorm weight renaming for checkpoint compatibility
5. Model saving/loading with compression modules
6. Mixed precision training support
7. Early stopping and better hyperparameters
"""

import os
import math
import argparse
import logging
import warnings
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"  # though we use local parquet, still good

# ---------- Offline settings ----------
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import orthogonal

from transformers import (
    AutoConfig, AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding,
    EarlyStoppingCallback, get_cosine_schedule_with_warmup,
    set_seed, PreTrainedModel, PretrainedConfig
)
from datasets import Dataset, DatasetDict
import evaluate

# ── Environment ─────────────────────────────────────────────
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

# ── 1. Compression Configuration ────────────────────────────
@dataclass
class CompressionConfig:
    """Configuration for token compression attention."""
    # Compression ratios
    k_early: int = 32      # First 4 layers
    k_mid: int = 16        # Middle 4 layers  
    k_late: int = 12       # Last 4 layers
    
    # Architecture
    progressive: bool = True
    use_orthogonal: bool = True
    use_gate: bool = True
    gate_bias: float = 0.0  # sigmoid(0) = 0.5 (equal weighting)
    
    # Training
    max_seq_len: int = 128
    attn_score_clamp: float = 50.0
    dropout: float = 0.1
    ortho_reg_weight: float = 0.001  # Only used when use_orthogonal=False
    
    def get_k_for_layer(self, layer_idx: int) -> int:
        """Get compression ratio for a specific layer."""
        if not self.progressive:
            return self.k_mid
        if layer_idx < 4:
            return self.k_early
        elif layer_idx < 8:
            return self.k_mid
        else:
            return self.k_late

# ── 2. Data Loading (Local Parquet) ────────────────────────
TASK_KEYS = {
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
    """Load GLUE dataset from local parquet files (raw, no tokenization)."""
    base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"
    task_path = os.path.join(base_path, task_name)
    if not os.path.exists(task_path):
        raise FileNotFoundError(f"Local path {task_path} not found")

    splits = {}
    for fname in os.listdir(task_path):
        if not fname.endswith(".parquet"):
            continue
        # split name: "train", "validation", etc. (works with both "train.parquet" and "train-00000-of-00001.parquet")
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

def get_num_labels_and_is_regression(task_name: str):
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

# ── 3. Orthogonal Projection Module ─────────────────────────
class OrthogonalProjection(nn.Module):
    """
    Learnable projection C ∈ ℝ^{n×k} with C^T C = I constraint.
    Uses PyTorch's orthogonal parametrization for exact orthogonality.
    """
    def __init__(self, n: int, k: int):
        super().__init__()
        self.n, self.k = n, k
        assert 1 <= k <= n, f"k={k} must be in [1, n={n}]"
        
        # Initialize with orthogonal matrix
        self.C_raw = nn.Parameter(torch.empty(n, k))
        nn.init.orthogonal_(self.C_raw)
        
        # Apply orthogonal constraint
        orthogonal(self, "C_raw")
    
    @property
    def C(self) -> torch.Tensor:
        """Returns the orthogonal matrix Q (n×k) where Q^T Q = I."""
        return self.C_raw  # Parametrization ensures orthogonality
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project x using C.
        
        Args:
            x: (..., n) or (..., n, d)
        Returns:
            (..., k) or (..., k, d)
        """
        if x.dim() == 2 and x.shape[-1] == self.n:
            # (batch, n) → (batch, k)
            return x @ self.C
        elif x.dim() == 3 and x.shape[-2] == self.n:
            # (batch, n, d) → (batch, k, d)
            # x: (B, n, d), C: (n, k)
            # x @ C: (B, d, k) → transpose to (B, k, d)
            return (x.transpose(-1, -2) @ self.C).transpose(-1, -2)
        else:
            raise ValueError(f"Unexpected input shape {x.shape}")




class TokenCompressedAttention(nn.Module):
    def __init__(self, config, comp_config: CompressionConfig, layer_idx: int):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.comp_config = comp_config
        self.layer_idx = layer_idx
        self.k = comp_config.get_k_for_layer(layer_idx)

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(comp_config.dropout)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        self.projection = OrthogonalProjection(comp_config.max_seq_len, self.k)

        if comp_config.use_gate:
            self.gate = nn.Parameter(torch.tensor(comp_config.gate_bias))
        else:
            self.gate = None

        self._ortho_loss = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,          # kept for signature compatibility
        output_attentions: bool = False,
        **kwargs
    ):
        B, N, D = hidden_states.size()
        scale = (self.head_dim ** -0.5)

        # 1) Q, K, V projections and reshape to (B, heads, N, head_dim)
        q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # 2) Attention scores (B, heads, N, N)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        scores = torch.clamp(scores, max=self.comp_config.attn_score_clamp)

        # 3) Apply attention mask (if provided)
        if attention_mask is not None:
            # BERT passes a mask of shape (B, 1, 1, N) with large negative for padding.
            # Add it directly.
            scores = scores + attention_mask

        # 4) Compression matrix C (N, k)
        C = self.projection.C[:N, :]

        # 5) Compress key dimension: (B, heads, N, N) @ (N, k) -> (B, heads, N, k)
        scores_comp = torch.matmul(scores, C)
        attn = F.softmax(scores_comp, dim=-1)
        attn = self.dropout(attn)

        # 6) Compress value dimension: (B, heads, N, head_dim) -> (B, heads, k, head_dim)
        v_comp = torch.matmul(v.transpose(-2, -1), C)   # (B, heads, head_dim, k)
        v_comp = v_comp.transpose(-2, -1)               # (B, heads, k, head_dim)

        # 7) Context: (B, heads, N, k) @ (B, heads, k, head_dim) -> (B, heads, N, head_dim)
        context = torch.matmul(attn, v_comp)

        # 8) Reshape back to (B, N, D)
        context = context.permute(0, 2, 1, 3).reshape(B, N, D)

        # 9) Gating (optional)
        if self.gate is not None:
            g = torch.sigmoid(self.gate)
            output = g * context + (1 - g) * hidden_states
        else:
            output = context

        # 10) Output projection (same as original BertSelfAttention)
        output = self.output(output)

        # 11) Orthogonality loss (soft constraint)
        self._ortho_loss = None
        if not self.comp_config.use_orthogonal:
            I_k = torch.eye(self.k, device=C.device, dtype=C.dtype)
            self._ortho_loss = F.mse_loss(C.T @ C, I_k)

        # 12) Return value(s) – IMPORTANT: always return a tuple of two items
        if output_attentions:
            return (output, attn)
        else:
            return (output, None)
            
        
    def get_ortho_loss(self):
            return self._ortho_loss




    # ── 5. Inject Compression into BERT ─────────────────────────
def inject_compression_into_layer(
    layer: nn.Module,
    comp_config: CompressionConfig,
    layer_idx: int,
    model_config
) -> TokenCompressedAttention:
    """Replace a BERT layer's self-attention with compressed attention."""
    
    # Create new compressed attention module
    new_attention = TokenCompressedAttention(
        config=model_config,
        comp_config=comp_config,
        layer_idx=layer_idx
    )
    
    # Copy weights from original attention
    old_attention = layer.attention.self
    
    # Copy Q/K/V weights
    new_attention.query.weight.data.copy_(old_attention.query.weight.data)
    new_attention.query.bias.data.copy_(old_attention.query.bias.data)
    
    new_attention.key.weight.data.copy_(old_attention.key.weight.data)
    new_attention.key.bias.data.copy_(old_attention.key.bias.data)
    
    new_attention.value.weight.data.copy_(old_attention.value.weight.data)
    new_attention.value.bias.data.copy_(old_attention.value.bias.data)
    
    # Copy output projection weights (CRITICAL!)
    new_attention.output.weight.data.copy_(layer.attention.output.dense.weight.data)
    new_attention.output.bias.data.copy_(layer.attention.output.dense.bias.data)
    
    # Replace the attention module
    layer.attention.self = new_attention
    
    return new_attention

def fix_layernorm_naming(model: nn.Module):
    """
    Fix LayerNorm parameter names from checkpoint format (beta/gamma) 
    to PyTorch format (bias/weight).
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.LayerNorm):
            # Check if parameters are named beta/gamma
            state_dict = dict(module.named_parameters())
            
            if 'beta' in state_dict and 'bias' not in state_dict:
                # Rename beta to bias
                bias_param = module._parameters.pop('beta')
                module.register_parameter('bias', bias_param)
            
            if 'gamma' in state_dict and 'weight' not in state_dict:
                # Rename gamma to weight
                weight_param = module._parameters.pop('gamma')
                module.register_parameter('weight', weight_param)

# ── 6. Main Model Wrapper ───────────────────────────────────
class BertWithTokenCompression(nn.Module):
    """
    BERT model with token-compressed attention layers.
    Compatible with HuggingFace Trainer.
    """
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_labels: int = 2,
        comp_config: Optional[CompressionConfig] = None,
        disable_compression: bool = False,
    ):
        super().__init__()
        
        self.comp_config = comp_config or CompressionConfig()
        self.disable_compression = disable_compression
        
        # Load base BERT model
        config = AutoConfig.from_pretrained(
            model_name, 
            num_labels=num_labels,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )
        
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Fix LayerNorm naming for checkpoint compatibility
        fix_layernorm_naming(self.bert)
        
        # Inject compression into all layers
        self.compression_modules = nn.ModuleList()
        if not disable_compression:
            for i, layer in enumerate(self.bert.bert.encoder.layer):
                new_attn = inject_compression_into_layer(
                    layer=layer,
                    comp_config=self.comp_config,
                    layer_idx=i,
                    model_config=config
                )
                self.compression_modules.append(new_attn)
        
        # Enable gradient checkpointing for memory efficiency
        self.bert.gradient_checkpointing_enable()
        
        log.info(f"Created model with {len(self.compression_modules)} compressed layers")
        log.info(f"Compression ratios: early={self.comp_config.k_early}, "
                f"mid={self.comp_config.k_mid}, late={self.comp_config.k_late}")
    
    def forward(self, **kwargs):
        """Forward pass through the model."""
        return self.bert(**kwargs)
    
    def get_ortho_loss(self) -> torch.Tensor:
        """Get total orthogonality loss from all compression modules."""
        total_loss = 0.0
        for module in self.compression_modules:
            loss = module.get_ortho_loss()
            if loss is not None:
                total_loss = total_loss + loss
        return torch.tensor(total_loss, device=self.bert.device)
    
    def save_pretrained(self, save_directory: str):
        """Save model and tokenizer."""
        self.bert.save_pretrained(save_directory)
    
    @property
    def device(self):
        return next(self.bert.parameters()).device

# ── 7. Custom Trainer with Orthogonality Loss ───────────────
# ── Custom Trainer with Orthogonality Loss ──────────────
class CompressionTrainer(Trainer):
    def __init__(self, ortho_weight: float = 0.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ortho_weight = ortho_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        task_loss = outputs.loss
        ortho_loss = model.get_ortho_loss()
        total_loss = task_loss + self.ortho_weight * ortho_loss
        return (total_loss, outputs) if return_outputs else total_loss
    
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from scipy.stats import pearsonr

# ── 8. Metrics ──────────────────────────────────────────────
def get_compute_metrics(task: str):
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
    
    
    # ── 8. Main ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Token‑Compressed Attention for GLUE")
    parser.add_argument("--task", type=str, default="sst2",
                        choices=["cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"])
    parser.add_argument("--model", type=str, default="bert-base-uncased")
    parser.add_argument("--output_dir", type=str, default="./compressed_results")
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_progressive", action="store_true")
    parser.add_argument("--no_gate", action="store_true")
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

    # Compression config
    comp_config = CompressionConfig(
        k_early=args.k_early,
        k_mid=args.k_mid,
        k_late=args.k_late,
        progressive=not args.no_progressive,
        use_gate=not args.no_gate,
        max_seq_len=args.max_len,
    )
    log.info(f"Task: {args.task} | Progressive: {comp_config.progressive}")
    log.info(f"k early/mid/late = {comp_config.k_early}/{comp_config.k_mid}/{comp_config.k_late}")

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    # Load raw dataset
    raw_datasets = load_glue_local(args.task)
    if 'train' not in raw_datasets:
        raise RuntimeError(f"No 'train' split in {args.task}. Found: {list(raw_datasets.keys())}")
    log.info(f"Raw dataset sizes: train={len(raw_datasets['train'])}, val={len(raw_datasets.get('validation', []))}")

    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Tokenization function (with explicit return)
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
        # Always add labels, even if it fails
        if is_regression:
            tokenized["labels"] = [float(label) for label in examples["label"]]
        else:
            tokenized["labels"] = examples["label"]
        return tokenized

    # Apply tokenization
    dataset = raw_datasets.map(preprocess, batched=True)

    # --- Sanity checks ---
    if len(dataset["train"]) == 0:
        raise RuntimeError("Training dataset became empty after tokenization!")
    # Check a sample to see if columns exist
    sample = dataset["train"][0]
    required = ["input_ids", "attention_mask", "labels"]
    missing = [col for col in required if col not in sample]
    if missing:
        raise RuntimeError(f"Missing columns after tokenization: {missing}")

    # Set format for PyTorch
    columns = ["input_ids", "attention_mask", "labels"]
    if args.task in ("mnli", "rte", "wnli", "mrpc", "qqp", "qnli"):
        columns.append("token_type_ids")
    dataset.set_format(type="torch", columns=columns)

    log.info(f"Tokenized dataset: train={len(dataset['train'])}, val={len(dataset.get('validation', []))}")

    # Build model
    model = BertWithTokenCompression(args.model, num_labels, comp_config)

    # Metric for best model (with 'eval_' prefix)
    if args.task == "cola":
        metric_name = "eval_matthews_correlation"
    elif args.task == "stsb":
        metric_name = "eval_pearsonr"
    elif args.task in ("mrpc", "qqp"):
        metric_name = "eval_f1"
    else:
        metric_name = "eval_accuracy"

    # Training arguments (remove deprecated overwrite_output_dir)
    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, args.task),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
            remove_unused_columns=False,   # ← add this line

        warmup_ratio=0.1,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        greater_is_better=True,
        fp16=args.fp16,
        save_total_limit=2,
        # remove_unused_columns=True,
        report_to="none",
    )

    ortho_weight = comp_config.ortho_reg_weight if not comp_config.use_orthogonal else 0.0
    trainer = CompressionTrainer(
        ortho_weight=ortho_weight,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        # tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(args.task),  # must return keys with 'eval_' prefix
    )

    log.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    log.info(f"Final eval: {eval_result}")
    trainer.save_model(os.path.join(args.output_dir, args.task))
    tokenizer.save_pretrained(os.path.join(args.output_dir, args.task))
    log.info("Done.")
if __name__ == "__main__":
    main()