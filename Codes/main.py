#!/usr/bin/env python3
"""
Master experiment runner for the Efficient Transformer Toolkit.

Usage:
    python run.py --config configs/sst2_galerkin.yaml
    python run.py --model distilbert-base-uncased --task sst2 \
                  --attn_type galerkin --rank 64 --method lora
"""

import argparse
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Parse command line and load full configuration
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--config", type=str, default=None, help="YAML config file")
parser.add_argument("--model", type=str, default="distilbert-base-uncased")
parser.add_argument("--task", "-t", type=str, default="sst2")
parser.add_argument("--attn_type", type=str, default="full")
parser.add_argument("--compressor", type=str, default="none")
parser.add_argument("--method", type=str, default="baseline")
parser.add_argument("--rank", type=int, default=32)
parser.add_argument("--svd_energy", type=float, default=0.95)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--output_dir", type=str, default="./experiments")
parser.add_argument("--exp_name", type=str, default=None)
parser.add_argument("--device", type=str, default="auto")
parser.add_argument("--fp16", action="store_true")
parser.add_argument("--save_visuals", action="store_true")
parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep")
args = parser.parse_args()

# Import framework after argument parsing to allow offline mode early
from framework.config import Config, load_config

# Load full configuration (YAML + CLI overrides)
cfg = load_config(args.config, cli_args=args)

# Override with any direct CLI values that the user provided
if args.attn_type != "full":
    cfg.compression.attn_type = args.attn_type
if args.rank != 32:
    cfg.compression.rank = args.rank
if args.batch_size != 16:
    cfg.training.batch_size = args.batch_size
if args.epochs != 5:
    cfg.training.epochs = args.epochs
if args.lr != 2e-5:
    cfg.training.learning_rate = args.lr
if args.seed != 42:
    cfg.training.seed = args.seed  # but TrainingConfig does not have seed? We'll handle seed separately
if args.fp16:
    cfg.hardware.enable_fp16 = True
if args.device != "auto":
    cfg.hardware.device = args.device

exp_name = args.exp_name or f"{cfg.compression.attn_type}_{cfg.method.method}_{args.task}_{args.seed}"
output_dir = Path(args.output_dir) / exp_name

# ---------------------------------------------------------------------------
# 2. Framework imports (offline mode already set via environment if needed)
# ---------------------------------------------------------------------------
import torch
import numpy as np
from transformers import set_seed

from framework.core.utils import set_seed as set_all_seeds
from framework.core.utils import get_device
from framework.data import (
    load_dataset,
    DataConfig,
    TaskType,
    preprocess_function,
    create_dataloader,
    ExperimentConfig,
)
from framework.models import load_model
from framework.training import (
    HuggingFaceTrainer,
    ResultsManager,
    get_task_metrics,
)
from framework.training.methods import get_method as get_method_fn
from framework.trainer.callbacks import (
    MemoryCallback,
    TimeCallback,
    GradientNormCallback,
)
from framework.results_saver import ResultsSaver, SaveConfig
from framework.tracking import ExperimentTracker, TrackingBackend

# Optional callbacks
try:
    from framework.callbacks.custom import EMACallback, SWACallback
except ImportError:
    EMACallback = SWACallback = None
try:
    from framework.callbacks.analysis import (
        SVDAttentionCallback,
        AttentionHeatmapCallback,
        GradientNormCallback as AnalysisGradCallback,
    )
except ImportError:
    SVDAttentionCallback = AttentionHeatmapCallback = AnalysisGradCallback = None

# ---------------------------------------------------------------------------
# 3. Setup
# ---------------------------------------------------------------------------
set_all_seeds(args.seed)
device = get_device(cfg.hardware.device)

# ---------------------------------------------------------------------------
# 4. Load model (with attention injection) and tokenizer
# ---------------------------------------------------------------------------
model, tokenizer = load_model(
    cfg.model.name,
    num_labels=cfg.model.num_labels,
    attention_type=cfg.compression.attn_type,
    attention_rank=cfg.compression.rank,
    variant=cfg.model.variants[0] if cfg.model.variants else "standard",
    cache_dir=cfg.model.cache_dir,
)

# ---------------------------------------------------------------------------
# 5. Load dataset and create dataloaders
# ---------------------------------------------------------------------------
exp_config = ExperimentConfig(dataset_name=args.task)
dataset = load_dataset(exp_config)

num_labels = cfg.model.num_labels
is_regression = (args.task == "stsb")
sent1_key, sent2_key = None, None
from framework.data.glue_loader import GLUE_TASK_KEYS
if args.task in GLUE_TASK_KEYS:
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

data_cfg = DataConfig(
    task_type=TaskType.SINGLE_CLASSIFICATION if sent2_key is None else TaskType.PAIR_CLASSIFICATION,
    max_length=cfg.data.max_seq_length,
    batch_size=cfg.training.batch_size,
    text_column_1=sent1_key,
    text_column_2=sent2_key,
    label_column="label",
)

def glue_preprocess(examples):
    return preprocess_function(examples, tokenizer, data_cfg)

tokenized = dataset.map(glue_preprocess, batched=True,
                       remove_columns=dataset["train"].column_names)
tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

train_loader = create_dataloader(tokenized["train"], tokenizer, data_cfg)
val_split = "validation" if "validation" in tokenized else "validation_matched"
val_loader = create_dataloader(tokenized[val_split], tokenizer, data_cfg) if val_split in tokenized else None

# ---------------------------------------------------------------------------
# 6. Apply training method (LoRA, Mixout, etc.)
# ---------------------------------------------------------------------------
if cfg.method.method != "baseline":
    method_fn = get_method_fn(cfg.method.method)
    if method_fn is None:
        raise ValueError(f"Unknown method: {cfg.method.method}")
    method_fn(model, cfg.compression)   # compression config contains method-specific params

# ---------------------------------------------------------------------------
# 7. Set up ResultsManager, Tracker, and Trainer
# ---------------------------------------------------------------------------
results = ResultsManager(
    experiment_name=exp_name,
    base_dir=str(args.output_dir),
    config=cfg.__dict__,
)

tracker = ExperimentTracker(
    backend=TrackingBackend.LOCAL,
    log_dir=str(output_dir / "tracking"),
    experiment_name=exp_name,
)

trainer = HuggingFaceTrainer(
    model=model,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset if val_loader else None,
    config=cfg.training.__dict__,
    results_manager=results,
    compute_metrics=get_task_metrics(args.task),
)

# Add basic callbacks
trainer.add_callback(MemoryCallback())
trainer.add_callback(TimeCallback())
trainer.add_callback(GradientNormCallback(log_every_n_steps=100))
if cfg.hardware.enable_fp16:
    trainer.args.fp16 = True

# Optional advanced callbacks
if args.attn_type != "full" and SVDAttentionCallback is not None:
    trainer.add_callback(SVDAttentionCallback(log_every_steps=500))
if args.save_visuals and AttentionHeatmapCallback is not None:
    trainer.add_callback(AttentionHeatmapCallback(output_dir=str(output_dir)))

# ---------------------------------------------------------------------------
# 8. Train
# ---------------------------------------------------------------------------
trainer.train()
eval_metrics = trainer.evaluate()

# ---------------------------------------------------------------------------
# 9. Save everything via ResultsSaver
# ---------------------------------------------------------------------------
saver_cfg = SaveConfig(
    save_model=True,
    save_plots=True,
    model_export_formats=["onnx"] if cfg.hardware.device == "cuda" else [],
    storage_backends=["local"],
)
saver = ResultsSaver(
    run_id=exp_name,
    base_dir=Path(args.output_dir),
    config=saver_cfg,
)
# Convert history to a list of dicts for the saver
history = [h for h in trainer.state.log_history] if hasattr(trainer, 'state') else []
saver.save_all(model, cfg.__dict__, history, eval_metrics, tokenizer)
saver.push_all()

print("=" * 60)
print(f"Experiment {exp_name} completed successfully!")
print(f"Results saved to {output_dir}")
print(f"Final metrics: {eval_metrics}")
print("=" * 60)