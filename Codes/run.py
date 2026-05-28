#!/usr/bin/env python3
"""
run.py – Universal experiment runner for the Efficient Transformer Toolkit.

Usage:
    python run.py --config configs/my_exp.yaml
    python run.py --model distilbert-base-uncased --task sst2 \
                  --attn_type galerkin --rank 64 --method lora
"""

import argparse, sys, os, json, time, torch
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Force offline mode – must happen before any HuggingFace import
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.getcwd(), "models")
os.environ["HF_HOME"] = CACHE_DIR
os.environ["TRANSFORMERS_CACHE"] = CACHE_DIR
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------------------------------------------------------
# Framework imports
# ---------------------------------------------------------------------------
from framework.config import Config, load_config
from framework.data import (
    load_dataset, DataConfig, TaskType, preprocess_function, create_dataloader,
    ExperimentConfig,
)
from framework.models import load_model
from framework.trainer import HuggingFaceTrainer, ResultsManager, get_task_metrics
from framework.trainer.methods import get_method as get_method_fn
from framework.results_saver import ResultsSaver, SaveConfig
from framework.tracking import ExperimentTracker, TrackingBackend
from framework.core.utils import set_seed, get_device

# Optional callbacks
try:
    from framework.callbacks.custom import EMACallback, SWACallback
except ImportError:
    EMACallback = SWACallback = None
try:
    from framework.callbacks.analysis import (
        SVDAttentionCallback, AttentionHeatmapCallback, GradientNormCallback
    )
except ImportError:
    SVDAttentionCallback = AttentionHeatmapCallback = GradientNormCallback = None

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
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
    parser.add_argument("--max_train_samples", type=int, default=-1)
    parser.add_argument("--save_visuals", action="store_true")
    args = parser.parse_args()

    # 1. Load unified configuration (YAML + CLI overrides)
    cfg = load_config(args.config, cli_args=args)

    # Apply specific overrides from CLI
    cfg.model.name = args.model
    cfg.compression.attn_type = args.attn_type
    if args.rank != 32:           cfg.compression.rank = args.rank
    if args.batch_size != 16:     cfg.training.batch_size = args.batch_size
    if args.epochs != 5:          cfg.training.epochs = args.epochs
    if args.lr != 2e-5:           cfg.training.learning_rate = args.lr
    if args.fp16:                 cfg.hardware.enable_fp16 = True
    if args.device != "auto":     cfg.hardware.device = args.device

    exp_name = args.exp_name or f"{cfg.compression.attn_type}_{cfg.method.method}_{args.task}_{args.seed}"
    output_dir = Path(args.output_dir) / exp_name

    # 2. Seed and device
    set_seed(args.seed)
    device = get_device(cfg.hardware.device)

    # 3. Load model (with attention injection) and tokenizer
    model, tokenizer = load_model(
        cfg.model.name,
        num_labels=cfg.model.num_labels,
        attention_type=cfg.compression.attn_type,
        attention_rank=cfg.compression.rank,
        cache_dir=cfg.model.cache_dir,
    )

    # 4. Load dataset and create dataloaders
    exp_config = ExperimentConfig(dataset_name=args.task)
    dataset = load_dataset(exp_config)

    num_labels = cfg.model.num_labels
    is_regression = (args.task == "stsb")
    from framework.data.glue_loader import GLUE_TASK_KEYS
    s1, s2 = GLUE_TASK_KEYS.get(args.task, (None, None))

    data_cfg = DataConfig(
        task_type=TaskType.SINGLE_CLASSIFICATION if s2 is None else TaskType.PAIR_CLASSIFICATION,
        max_length=cfg.data.max_seq_length,
        batch_size=cfg.training.batch_size,
        text_column_1=s1, text_column_2=s2, label_column="label",
    )

    def preprocess(examples):
        return preprocess_function(examples, tokenizer, data_cfg)

    tokenized = dataset.map(preprocess, batched=True,
                           remove_columns=dataset["train"].column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    train_loader = create_dataloader(tokenized["train"], tokenizer, data_cfg)
    val_split = "validation" if "validation" in tokenized else "validation_matched"
    val_loader = create_dataloader(tokenized[val_split], tokenizer, data_cfg) if val_split in tokenized else None

    # 5. Apply training method
    if cfg.method.method != "baseline":
        method_fn = get_method_fn(cfg.method.method)
        if method_fn:
            method_fn(model, cfg.compression)

    # 6. Set up ResultsManager, Tracker, Trainer
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
    # Basic callbacks
    trainer.add_callback(tracker.get_callback())   # assume tracker provides a callback
    if cfg.hardware.enable_fp16:
        trainer.args.fp16 = True
    if SVDAttentionCallback:
        trainer.add_callback(SVDAttentionCallback(log_every_steps=500))
    if args.save_visuals and AttentionHeatmapCallback:
        trainer.add_callback(AttentionHeatmapCallback(output_dir=str(output_dir)))

    # 7. Train
    trainer.train()
    eval_metrics = trainer.evaluate()

    # 8. Save everything
    saver_cfg = SaveConfig(
        save_model=True,
        save_plots=True,
        model_export_formats=["onnx"] if device.type == "cuda" else [],
        storage_backends=["local"],
    )
    saver = ResultsSaver(
        run_id=exp_name,
        base_dir=Path(args.output_dir),
        config=saver_cfg,
    )
    history = getattr(trainer.state, "log_history", [])
    saver.save_all(model, cfg.__dict__, history, eval_metrics, tokenizer)
    saver.push_all()

    print(f"Experiment {exp_name} completed. Results in {output_dir}")
    print(f"Final metrics: {eval_metrics}")

if __name__ == "__main__":
    main()