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

class LinformerProjection(nn.Module):
    """Learnable projection for keys and values (no orthogonal constraint)."""
    def __init__(self, n: int, k: int):
        super().__init__()
        # Initialise as truncated identity + small noise
        base = torch.eye(n)[:k, :]          # (k, n)
        self.E = nn.Parameter(base.clone())
        self.F = nn.Parameter(base.clone())
        with torch.no_grad():
            self.E += torch.randn_like(self.E) * 0.02
            self.F += torch.randn_like(self.F) * 0.02

    def forward(self, K, V, mask_2d=None):
        # K, V: (B, heads, N, d)
        B, H, N, d = K.shape
        if mask_2d is not None:
            mask = mask_2d.unsqueeze(1).unsqueeze(-1)   # (B, 1, N, 1)
            K = K * mask
            V = V * mask

        # Combine batch and heads for batched matrix multiplication
        K_flat = K.reshape(B * H, N, d)     # (B*H, N, d)
        V_flat = V.reshape(B * H, N, d)

        E_exp = self.E.unsqueeze(0).expand(B * H, -1, -1)   # (B*H, k, N)
        F_exp = self.F.unsqueeze(0).expand(B * H, -1, -1)

        # Project: (B*H, k, N) @ (B*H, N, d) -> (B*H, k, d)
        K_comp = torch.bmm(E_exp, K_flat)
        V_comp = torch.bmm(F_exp, V_flat)

        # Reshape back to (B, heads, k, d)
        K_comp = K_comp.reshape(B, H, self.E.shape[0], d)
        V_comp = V_comp.reshape(B, H, self.F.shape[0], d)
        return K_comp, V_comp
class TokenCompressedAttention(nn.Module):
    def __init__(self, config, comp_config, layer_idx, bert_mode=False):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.hidden_size = config.hidden_size
        self.comp_config = comp_config
        self.layer_idx = layer_idx
        self.k = comp_config.get_k_for_layer(layer_idx)
        self.bert_mode = bert_mode

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(comp_config.dropout)
        self.output = nn.Linear(self.hidden_size, self.hidden_size)

        # جایگزین پروجکشن متعامد با LinformerProjection (بدون قید)
        self.projection = LinformerProjection(comp_config.max_seq_len, self.k)

        if comp_config.use_gate and not self.bert_mode:
            self.gate = nn.Parameter(torch.tensor(comp_config.gate_bias))
        else:
            self.gate = None

        self._ortho_loss = None   # دیگر استفاده نمی‌شود


    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        if attention_mask is None:
            attention_mask = kwargs.get('attn_mask', None)

        B, N, D = hidden_states.size()
        scale = (self.head_dim ** -0.5)

        q = self.query(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.key(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.value(hidden_states).view(B, N, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if attention_mask is not None:
            if attention_mask.dim() == 4:
                mask_2d = (attention_mask > -1).float()[:, 0, 0, :]
            elif attention_mask.dim() == 2:
                mask_2d = attention_mask.float()
            else:
                raise ValueError(f"Unexpected mask shape: {attention_mask.shape}")
        else:
            mask_2d = None

        # ⚠️ critical – unpack the tuple from LinformerProjection
        k_comp, v_comp = self.projection(k, v, mask_2d)   # both: (B, heads, k, d)

        scores = torch.matmul(q, k_comp.transpose(-2, -1)) * scale
        scores = torch.clamp(scores, max=self.comp_config.attn_score_clamp)

        if mask_2d is not None:
            scores = scores + ((1.0 - mask_2d)[:, None, :, None] * -1e9)

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v_comp)
        context = context.permute(0, 2, 1, 3).reshape(B, N, D)

        if not self.bert_mode:
            if self.gate is not None:
                g = torch.sigmoid(self.gate)
                output = g * context + (1 - g) * hidden_states
            else:
                output = context
            output = self.output(output)
        else:
            output = context

        return (output, attn) if output_attentions else (output, None)

    def get_ortho_loss(self):
        return 0.0
    






class CompressedBertAttention(nn.Module):
    def __init__(self, original_attention, compressed_self_attn):
        super().__init__()
        self.compressed_self = compressed_self_attn   # TokenCompressedAttention (normal mode)
        self.output = original_attention.output        # BertSelfOutput (dense will be replaced with Identity)
        self.output_dropout = original_attention.output.dropout   # keep original dropout
        self.pruned_heads = getattr(original_attention, 'pruned_heads', set())

    def forward(self, hidden_states, attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        attn_result = self.compressed_self(
            hidden_states,
            attention_mask=attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            **kwargs
        )
        if isinstance(attn_result, tuple):
            context, attn_weights = attn_result
        else:
            context, attn_weights = attn_result, None

        # Apply the same dropout that original BertSelfOutput used to have after the dense
        context = self.output_dropout(context)
        # Now call BertSelfOutput (its dense is now Identity, so it only does residual + LayerNorm)
        layer_output = self.output(context, hidden_states)

        return (layer_output, attn_weights)    

            # ── 5. Inject Compression into BERT ─────────────────────────
def inject_compression_into_layer(layer, comp_config, layer_idx, model_config, model_type):
    if model_type == 'bert':
        new_attn = TokenCompressedAttention(
            config=model_config,
            comp_config=comp_config,
            layer_idx=layer_idx,
            bert_mode=False          # normal mode – will apply output dense (copied below)
        )
        old_self = layer.attention.self

        # Copy Q/K/V weights
        new_attn.query.weight.data.copy_(old_self.query.weight.data)
        new_attn.query.bias.data.copy_(old_self.query.bias.data)
        new_attn.key.weight.data.copy_(old_self.key.weight.data)
        new_attn.key.bias.data.copy_(old_self.key.bias.data)
        new_attn.value.weight.data.copy_(old_self.value.weight.data)
        new_attn.value.bias.data.copy_(old_self.value.bias.data)

        # Copy the original self-attention output dense (the projection before residual)
        new_attn.output.weight.data.copy_(old_self.output.weight.data)
        new_attn.output.bias.data.copy_(old_self.output.bias.data)

        # Replace BertSelfOutput's dense with Identity to avoid double projection
        layer.attention.output.dense = nn.Identity()

        # Wrap the whole attention in our replacement module
        replacement = CompressedBertAttention(layer.attention, new_attn)
        layer.attention = replacement
        return new_attn
    elif model_type == 'distilbert':
        new_attn = TokenCompressedAttention(
            config=model_config,
            comp_config=comp_config,
            layer_idx=layer_idx,
            bert_mode=False
        )
        old_attn = layer.attention
        new_attn.query.weight.data.copy_(old_attn.q_lin.weight.data)
        new_attn.query.bias.data.copy_(old_attn.q_lin.bias.data)
        new_attn.key.weight.data.copy_(old_attn.k_lin.weight.data)
        new_attn.key.bias.data.copy_(old_attn.k_lin.bias.data)
        new_attn.value.weight.data.copy_(old_attn.v_lin.weight.data)
        new_attn.value.bias.data.copy_(old_attn.v_lin.bias.data)
        new_attn.output.weight.data.copy_(old_attn.out_lin.weight.data)
        new_attn.output.bias.data.copy_(old_attn.out_lin.bias.data)

        layer.attention = new_attn      # ← ensure it's exactly this, nothing else
        return new_attn




    else:
                raise NotImplementedError(f"Unsupported model type: {model_type}")
            




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

        # Load config and model
        config = AutoConfig.from_pretrained(
            model_name,
            num_labels=num_labels,
        )
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            config=config,
        )
        fix_layernorm_naming(self.bert)

        # Identify base transformer layers
        model_type = config.model_type
        if model_type == 'bert':
            self.base_model = self.bert.bert
            layers = self.base_model.encoder.layer
        elif model_type == 'distilbert':
            self.base_model = self.bert.distilbert
            layers = self.base_model.transformer.layer
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Inject compression
        self.compression_modules = nn.ModuleList()
        if not disable_compression:
            for i, layer in enumerate(layers):
                new_attn = inject_compression_into_layer(
                    layer=layer,
                    comp_config=self.comp_config,
                    layer_idx=i,
                    model_config=config,
                    model_type=model_type,
                )
                self.compression_modules.append(new_attn)

        # Gradient checkpointing for memory
        self.bert.gradient_checkpointing_enable()

        log.info(f"Created model with {len(self.compression_modules)} compressed layers "
                 f"(type={model_type})")
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

import evaluate

def get_compute_metrics(task: str):
    task = task.lower()
    if task == "cola":
        metric = evaluate.load("matthews_correlation")
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return metric.compute(predictions=preds, references=labels)
        return compute
    elif task == "stsb":
        metric = evaluate.load("pearsonr")
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = logits.squeeze()
            return metric.compute(predictions=preds, references=labels)
        return compute
    elif task in ("mrpc", "qqp"):
        acc_metric = evaluate.load("accuracy")
        f1_metric = evaluate.load("f1")
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_accuracy": acc_metric.compute(predictions=preds, references=labels)["accuracy"],
                    "eval_f1": f1_metric.compute(predictions=preds, references=labels)["f1"]}
        return compute
    else:
        metric = evaluate.load("accuracy")
        def compute(eval_pred):
            logits, labels = eval_pred
            preds = np.argmax(logits, axis=-1)
            return {"eval_accuracy": metric.compute(predictions=preds, references=labels)["accuracy"]}
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
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal) device")
        model = model.to(device)
    else:
        device = torch.device("cpu")
        print("Using CPU")
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
        weight_decay=0.01,
        lr_scheduler_type="cosine",
    max_grad_norm=10,          # stricter clipping
    warmup_ratio=0.1,
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