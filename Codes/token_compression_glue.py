#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Enhanced GLUE with Mixed Compression Strategies
SVD + LoRA + Adaptive Rank + Gradient Optimization
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
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
from scipy.linalg import svd
import warnings
warnings.filterwarnings('ignore')

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"  # MPS memory optimization

# ============================================================================
# Global Configuration
# ============================================================================
GLUE_TASK_KEYS = {
    "cola": ("sentence", None), "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"), "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"), "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"), "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}

# ============================================================================
# Advanced Compression Modules
# ============================================================================

class AdaptiveSVDLinear(nn.Module):
    """
    SVD Linear with adaptive rank and gradient optimization
    Implements gradient checkpointing and memory-efficient forward pass
    """
    def __init__(self, in_features: int, out_features: int, 
                 max_rank: int, bias: bool = True,
                 use_checkpoint: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.use_checkpoint = use_checkpoint
        
        # Learnable rank selection
        self.rank_importance = nn.Parameter(torch.ones(max_rank))
        
        # Low-rank factors with initialization from SVD
        self.U = nn.Parameter(torch.empty(out_features, max_rank))
        self.V = nn.Parameter(torch.empty(max_rank, in_features))
        
        # Residual connection parameters (like LoRA)
        self.A = nn.Parameter(torch.zeros(max_rank, in_features))
        self.B = nn.Parameter(torch.zeros(out_features, max_rank))
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        self._effective_rank = max_rank
        
    def reset_parameters(self):
        # Orthogonal initialization for better gradient flow
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V.T)
        
        # Initialize A and B with small values (LoRA style)
        nn.init.kaiming_uniform_(self.A, a=np.sqrt(5))
        nn.init.kaiming_uniform_(self.B, a=np.sqrt(5))
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def compute_effective_rank(self, threshold: float = 0.01):
        """Dynamically adjust rank based on importance scores"""
        importance = F.softmax(self.rank_importance, dim=0)
        cumulative = torch.cumsum(importance, dim=0)
        effective_rank = torch.sum(cumulative < (1 - threshold)).item() + 1
        self._effective_rank = min(max(effective_rank, 4), self.max_rank)
        return self._effective_rank
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and self.use_checkpoint:
            return self._forward_checkpoint(x)
        else:
            return self._forward_impl(x)
    
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        # Compute effective rank
        r = self.compute_effective_rank()
        
        # Base low-rank projection
        U_r = self.U[:, :r]
        V_r = self.V[:r, :]
        
        # Residual adaptation (LoRA-like)
        A_r = self.A[:r, :]
        B_r = self.B[:, :r]
        
        # Fused computation: (U @ V + alpha * B @ A) @ x
        # This reduces memory by avoiding intermediate storage
        
        # Compute Vx first (smaller intermediate)
        Vx = F.linear(x, V_r.T)  # shape: (batch, r)
        
        # Compute Ax for residual
        Ax = F.linear(x, A_r.T)  # shape: (batch, r)
        
        # Combine: U(Vx) + alpha * B(Ax)
        base_out = F.linear(Vx, U_r.T)  # shape: (batch, out_features)
        residual_out = self.alpha * F.linear(Ax, B_r.T)  # shape: (batch, out_features)
        
        output = base_out + residual_out
        
        if self.bias is not None:
            output = output + self.bias
        
        return output
    
    def _forward_checkpoint(self, x: torch.Tensor) -> torch.Tensor:
        """Gradient checkpointing for memory efficiency"""
        return torch.utils.checkpoint.checkpoint(
            self._forward_impl, x, 
            use_reentrant=False,
            preserve_rng_state=False
        )
    
    @property
    def weight(self):
        """Return full weight matrix for compatibility"""
        r = self._effective_rank
        return self.U[:, :r] @ self.V[:r, :] + self.alpha * self.B[:, :r] @ self.A[:r, :]
    
    @property
    def parameter_count(self):
        """Return number of parameters"""
        return (self.U.numel() + self.V.numel() + 
                self.A.numel() + self.B.numel() +
                (self.bias.numel() if self.bias is not None else 0) +
                self.rank_importance.numel() + 1)

class MixedCompressionBert(nn.Module):
    """
    BERT with mixed compression strategies per layer type
    """
    def __init__(self, config, compression_config: Dict):
        super().__init__()
        self.config = config
        self.compression_config = compression_config
        
        # Initialize original BERT
        self.bert = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", 
            config=config,
            local_files_only=True
        ).bert
        
        # Compression ratios per layer type
        self.layer_configs = {
            "attention": {"max_rank": 32, "use_lora": True},
            "ffn": {"max_rank": 64, "use_lora": False},
            "output": {"max_rank": 128, "use_lora": True},
        }
        
        self.compress_model()
        
        # Add classifier (uncompressed)
        self.classifier = nn.Linear(
            config.hidden_size, 
            config.num_labels
        )
        
    def compress_model(self):
        """Apply mixed compression to BERT layers"""
        for name, module in self.bert.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            
            # Skip classifier and embedding layers
            if "embeddings" in name or "classifier" in name or "pooler" in name:
                continue
            
            # Determine layer type and compression strategy
            layer_type = self._get_layer_type(name)
            config = self.layer_configs.get(layer_type, {"max_rank": 64, "use_lora": True})
            
            # Get original weight and compute SVD
            W_orig = module.weight.data.cpu().numpy()
            out_dim, in_dim = W_orig.shape
            
            # Adaptive rank based on singular values
            U, S, Vt = svd(W_orig, full_matrices=False)
            explained_variance = np.cumsum(S) / np.sum(S)
            target_variance = 0.95  # Keep 95% of variance
            
            # Find rank that preserves variance
            rank = np.searchsorted(explained_variance, target_variance) + 1
            rank = min(rank, config["max_rank"], out_dim, in_dim)
            
            # Initialize compressed layer
            compressed = AdaptiveSVDLinear(
                in_features=in_dim,
                out_features=out_dim,
                max_rank=rank,
                bias=module.bias is not None,
                use_checkpoint=self.compression_config.get("use_checkpoint", True)
            )
            
            # Initialize with SVD factors
            with torch.no_grad():
                compressed.U[:, :rank] = torch.tensor(U[:, :rank], dtype=module.weight.dtype)
                compressed.V[:rank, :] = torch.tensor(np.diag(S[:rank]) @ Vt[:rank, :], 
                                                     dtype=module.weight.dtype)
                if module.bias is not None:
                    compressed.bias.copy_(module.bias)
            
            # Replace the layer
            parent = self._get_parent_module(name)
            setattr(parent, name.split('.')[-1], compressed)
            
            print(f"Compressed {name} | Type: {layer_type} | Rank: {rank}")
    
    def _get_layer_type(self, name: str) -> str:
        """Determine layer type for compression strategy"""
        if "attention" in name:
            if "output" in name:
                return "attention"
            return "attention"
        elif "intermediate" in name:
            return "ffn"
        elif "output" in name and "dense" in name:
            return "output"
        return "ffn"
    
    def _get_parent_module(self, name: str) -> nn.Module:
        """Get parent module for layer replacement"""
        parts = name.split('.')
        parent = self.bert
        for part in parts[:-1]:
            parent = getattr(parent, part)
        return parent
    
    def forward(self, input_ids, attention_mask=None, token_type_ids=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)
        
        loss = None
        if labels is not None:
            if self.config.num_labels == 1:
                loss = F.mse_loss(logits.squeeze(), labels.squeeze())
            else:
                loss = F.cross_entropy(logits, labels)
        
        return (loss, logits) if loss is not None else (logits,)

# ============================================================================
# Gradient Optimization
# ============================================================================

class GradientOptimizer:
    """
    Advanced gradient optimization for compressed models
    - Gradient accumulation
    - Dynamic gradient clipping
    - Mixed precision training
    """
    def __init__(self, model, optimizer, accumulation_steps=4, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.step_count = 0
        
    def step(self, loss):
        # Scale loss for gradient accumulation
        loss = loss / self.accumulation_steps
        
        # Backward pass
        loss.backward()
        
        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.max_grad_norm
            )
            
            # Update weights
            self.optimizer.step()
            self.optimizer.zero_grad()
            
    def zero_grad(self):
        self.optimizer.zero_grad()

# ============================================================================
# Main Pipeline Functions
# ============================================================================

def load_glue_local(task_name: str) -> DatasetDict:
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
        
        try:
            ds = Dataset.from_parquet([file_path])
            splits[split] = ds
        except Exception as e:
            print(f"Warning: Could not load {file_path}: {e}")
            continue
    
    return DatasetDict(splits)

def get_task_config(task_name: str):
    task_name = task_name.lower()
    
    task_configs = {
        "cola": {"num_labels": 2, "regression": False, "metric": "matthews_correlation"},
        "sst2": {"num_labels": 2, "regression": False, "metric": "accuracy"},
        "mrpc": {"num_labels": 2, "regression": False, "metric": "f1"},
        "qqp": {"num_labels": 2, "regression": False, "metric": "f1"},
        "stsb": {"num_labels": 1, "regression": True, "metric": "pearson_spearman"},
        "mnli": {"num_labels": 3, "regression": False, "metric": "accuracy"},
        "qnli": {"num_labels": 2, "regression": False, "metric": "accuracy"},
        "rte": {"num_labels": 2, "regression": False, "metric": "accuracy"},
        "wnli": {"num_labels": 2, "regression": False, "metric": "accuracy"},
    }
    
    return task_configs[task_name]

def compute_metrics_builder(task_name):
    task_config = get_task_config(task_name)
    metric_name = task_config["metric"]
    
    if task_name == "stsb":
        metric = evaluate.load("glue", "stsb")
    else:
        metric = evaluate.load("glue", task_name)
    
    def compute_metrics(p):
        predictions = p.predictions
        labels = p.label_ids
        
        if task_config["regression"]:
            predictions = predictions.squeeze()
        else:
            predictions = np.argmax(predictions, axis=1)
        
        results = metric.compute(predictions=predictions, references=labels)
        
        # Special handling for STS-B
        if task_name == "stsb" and "pearson" in results and "spearmanr" in results:
            results["pearson_spearman"] = (results["pearson"] + results["spearmanr"]) / 2.0
        
        return results
    
    return compute_metrics

# ============================================================================
# Memory Efficient Trainer
# ============================================================================

class EfficientTrainer(Trainer):
    """
    Custom trainer with memory optimizations and gradient handling
    """
    def __init__(self, *args, gradient_accumulation=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.gradient_accumulation = gradient_accumulation
        self.gradient_optimizer = None
        
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation with gradient accumulation
        """
        outputs = model(**inputs)
        loss = outputs[0]
        
        # Apply gradient accumulation scaling
        if self.gradient_accumulation > 1:
            loss = loss / self.gradient_accumulation
        
        return (loss, outputs) if return_outputs else loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with gradient optimization
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        
        if self.args.n_gpu > 1:
            loss = loss.mean()
        
        # Scale loss if we're using gradient accumulation
        if self.gradient_accumulation > 1:
            loss = loss / self.gradient_accumulation
        
        loss.backward()
        
        return loss.detach()

# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Enhanced GLUE with Mixed Compression")
    
    # Task and model arguments
    parser.add_argument("--task", type=str, required=True,
                       choices=["cola", "sst2", "mrpc", "rte", "stsb", 
                               "mnli", "qnli", "qqp", "wnli"])
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    
    # Compression arguments
    parser.add_argument("--max_rank", type=int, default=64)
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--use_checkpoint", action="store_true", default=True)
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)  # Smaller for MPS
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=100)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # Optimization arguments
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=500)
    
    # System arguments
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./compressed_model")
    parser.add_argument("--fp16", action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load task configuration
    task_config = get_task_config(args.task)
    
    print(f"🚀 Starting Enhanced Compression Pipeline")
    print(f"Task: {args.task}")
    print(f"Compression: Mixed SVD+LoRA | Max Rank: {args.max_rank}")
    print(f"Training: {args.epochs} epochs | LR: {args.learning_rate}")
    print(f"Batch Size: {args.batch_size} | Gradient Accumulation: {args.gradient_accumulation}")
    
    # Load tokenizer and base config
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, local_files_only=True)
    
    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=task_config["num_labels"],
        local_files_only=True
    )
    
    # Compression configuration
    compression_config = {
        "max_rank": args.max_rank,
        "use_lora": args.use_lora,
        "use_checkpoint": args.use_checkpoint,
    }
    
    # Create compressed model
    print("\n🔧 Building compressed model...")
    model = MixedCompressionBert(config, compression_config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Original BERT parameters for comparison
    orig_model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config, local_files_only=True
    )
    orig_params = sum(p.numel() for p in orig_model.parameters())
    
    print(f"\n📊 Parameter Statistics:")
    print(f"  Original BERT: {orig_params:,}")
    print(f"  Compressed:    {total_params:,}")
    print(f"  Trainable:     {trainable_params:,}")
    print(f"  Compression:   {total_params/orig_params:.2%} of original")
    print(f"  Reduction:     {1 - total_params/orig_params:.2%}")
    
    # Load dataset
    print(f"\n📁 Loading {args.task} dataset...")
    datasets = load_glue_local(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]
    
    def preprocess_function(examples):
        if sent2_key is None:
            return tokenizer(
                examples[sent1_key],
                truncation=True,
                padding='max_length',
                max_length=args.max_seq_length
            )
        else:
            return tokenizer(
                examples[sent1_key],
                examples[sent2_key],
                truncation=True,
                padding='max_length',
                max_length=args.max_seq_length
            )
    
        # Preprocess datasets
    columns_to_remove = [
        col for col in datasets["train"].column_names 
        if col not in ["label"]
    ]

    encoded_datasets = datasets.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
    )
        
    # Rename label column for GLUE
    encoded_datasets = encoded_datasets.rename_column("label", "labels")
    
    # Prepare training and evaluation datasets
    train_dataset = encoded_datasets["train"]
    eval_key = "validation_matched" if args.task == "mnli" else "validation"
    eval_dataset = encoded_datasets[eval_key]
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        # overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_grad_norm=args.max_grad_norm,
        logging_dir=os.path.join(args.output_dir, "logs"),
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model=f"eval_{task_config['metric']}",
        greater_is_better=True,
        save_total_limit=2,
        seed=args.seed,
        fp16=args.fp16,
        dataloader_drop_last=True,
        remove_unused_columns=True,
        label_names=["labels"],
    )
    
    # Create compute metrics function
    compute_metrics = compute_metrics_builder(args.task)
    
    # Initialize trainer
    print(f"\n🎯 Starting training...")
    
    trainer = EfficientTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        # tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    
    # Train the model
    train_result = trainer.train()
    
    # Evaluate the model
    print(f"\n📈 Evaluating model...")
    eval_results = trainer.evaluate()
    
    print(f"\n✅ Training completed!")
    print(f"Final evaluation results:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Save the model
    print(f"\n💾 Saving model to {args.output_dir}...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training metrics
    metrics_file = os.path.join(args.output_dir, "training_metrics.txt")
    with open(metrics_file, "w") as f:
        f.write(f"Task: {args.task}\n")
        f.write(f"Original parameters: {orig_params:,}\n")
        f.write(f"Compressed parameters: {total_params:,}\n")
        f.write(f"Compression ratio: {total_params/orig_params:.2%}\n")
        f.write(f"Training loss: {train_result.training_loss:.4f}\n")
        f.write(f"Evaluation results:\n")
        for key, value in eval_results.items():
            f.write(f"  {key}: {value:.4f}\n")
    
    print(f"\n🎉 Done! Model saved to {args.output_dir}")
    print(f"📊 Metrics saved to {metrics_file}")

if __name__ == "__main__":
    main()
