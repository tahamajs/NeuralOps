#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning & Utility Functions
"""
import sys
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import Trainer, TrainerCallback, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from framework.experiment.config import ExperimentConfig
from framework.experiment.data import load_and_preprocess_data
from framework.experiment.model import load_model
from framework.experiment.train import setup_trainer
from framework.utils.seed import set_all_seeds
from framework.utils.device import get_device
from framework.compression_engine.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# Django callback (for progress/log/metrics during training)
# ----------------------------------------------------------------------
class DjangoCallback(TrainerCallback):
    def __init__(self, progress_cb, log_cb, final_cb, stop_event=None):
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.final_cb = final_cb
        self.stop_event = stop_event
        self._total_steps = None

    def on_train_begin(self, args, state, control, **kwargs):
        self._total_steps = state.max_steps

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.log_cb:
            self.log_cb(state.global_step, logs or {})

    def on_step_end(self, args, state, control, **kwargs):
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
        if self.progress_cb and self._total_steps:
            pct = (state.global_step / self._total_steps) * 100
            self.progress_cb(state.global_step, self._total_steps, pct)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if self.log_cb and metrics:
            self.log_cb(state.global_step, metrics)

    def on_train_end(self, args, state, control, **kwargs):
        # final callback is called manually after evaluation
        pass


# ----------------------------------------------------------------------
# Training entry point (used by Django runner)
# ----------------------------------------------------------------------
def run_training(config, stop_event=None,
                 progress_callback=None, log_callback=None, final_callback=None):
    """
    Full training pipeline.
    config: dict or ExperimentConfig instance.
    Returns evaluation metrics dict after training.
    """
    # Normalise config
    if isinstance(config, dict):
        cfg = ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    # Add GLUE base path from Django settings if available
    try:
        from django.conf import settings
        cfg.glue_base_path = getattr(settings, 'GLUE_DATA_PATH', cfg.glue_base_path)
    except ImportError:
        pass
    if not hasattr(cfg, 'glue_base_path') or not cfg.glue_base_path:
        cfg.glue_base_path = "/Users/tahamajs/Documents/uni/Project/datasets/glue"  # fallback

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"

    set_all_seeds(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data & model
    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)
    model = load_model(cfg, num_labels, device)
    model.to(device)

    # Build trainer
    trainer = setup_trainer(model, tokenizer, dataset, cfg, None, logger)

    # Inject Django callbacks
    cb = DjangoCallback(progress_callback, log_callback, final_callback, stop_event)
    trainer.add_callback(cb)

    # Train
    logger.info("Starting training...")
    train_result = trainer.train()
    eval_result = trainer.evaluate()

    if final_callback:
        final_callback(eval_result, trainer.state.log_history)

    logger.info("Training completed successfully.")
    return eval_result


# ----------------------------------------------------------------------
# Evaluation runner (standalone)
# ----------------------------------------------------------------------
def run_evaluation(config, model_path):
    """
    Evaluate a trained model on the specified GLUE task.
    config: dict or ExperimentConfig (must contain task_name or task)
    model_path: directory containing the saved model (e.g., 'pytorch_model.bin')
    Returns dict with evaluation metrics.
    """
    if isinstance(config, dict):
        cfg = ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    set_all_seeds(cfg.seed)
    device = get_device()

    # Load dataset (evaluation only, no training)
    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)

    # Load model from the saved path
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception:
        logger.warning(f"Could not load model from {model_path}, falling back to base model")
        model = load_model(cfg, num_labels, device)

    model.to(device)

    # Create a simple trainer just for evaluation (no training args needed)
    from transformers import TrainingArguments, DataCollatorWithPadding
    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=cfg.batch_size * 2,
        remove_unused_columns=False,
        report_to="none",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
    )

    eval_result = trainer.evaluate()
    # Normalise metric keys (e.g., "eval_accuracy" -> "accuracy")
    cleaned = {k.replace("eval_", ""): v for k, v in eval_result.items()}
    return cleaned


# ----------------------------------------------------------------------
# Compression runner (simplified, using COMPRESSOR_REGISTRY)
# ----------------------------------------------------------------------
def run_compression(config, output_dir):
    """
    Compress a trained model using the selected compressor from config.
    config: dict or ExperimentConfig (must contain compressor, compression_rank, etc.)
    output_dir: where to save compressed model
    Returns dict with compression statistics.
    """
    if isinstance(config, dict):
        cfg = ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    compressor_name = cfg.compressor or "svd"  # default
    if compressor_name not in COMPRESSOR_REGISTRY:
        raise ValueError(f"Unknown compressor: {compressor_name}")

    set_all_seeds(cfg.seed)
    device = get_device()

    # Load base model (we'll compress it)
    model = load_model(cfg, num_labels=2, device=device)  # num_labels might be overridden by model config
    model.to(device)

    # Instantiate compressor from registry
    compressor_cls = COMPRESSOR_REGISTRY[compressor_name]
    compressor = compressor_cls(model, rank=cfg.compression_rank)

    # Apply compression
    logger.info(f"Applying compression: {compressor_name}")
    compressed_model = compressor.compress()

    # Save compressed model
    os.makedirs(output_dir, exist_ok=True)
    compressed_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.save_pretrained(output_dir)

    # Compute size stats
    original_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    compressed_bytes = sum(p.numel() * p.element_size() for p in compressed_model.parameters())
    reduction_ratio = original_bytes / compressed_bytes if compressed_bytes > 0 else 1

    stats = {
        "compressor": compressor_name,
        "original_size_mb": round(original_bytes / (1024**2), 2),
        "compressed_size_mb": round(compressed_bytes / (1024**2), 2),
        "reduction_ratio": round(reduction_ratio, 2),
        "compression_rank": cfg.compression_rank,
    }
    logger.info(f"Compression complete: {stats}")
    return stats


# ----------------------------------------------------------------------
# CLI entry point
# ----------------------------------------------------------------------

def main():
    cfg = build_config_from_args()

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"
    # Ensure necessary attributes exist for the logger
    if not hasattr(cfg, 'log_base_dir'):
        cfg.log_base_dir = cfg.log_dir
    if not hasattr(cfg, 'use_timestamp'):
        cfg.use_timestamp = True
    if not hasattr(cfg, 'run_id'):
        cfg.run_id = None

    # Build LoggingConfig from ExperimentConfig
    log_cfg = LoggingConfig(
        log_base_dir=cfg.log_base_dir,
        experiment_name=exp_name,
        run_id=cfg.run_id,
        use_timestamp=cfg.use_timestamp,
    )
    logger = setup_logging(log_cfg)

    tracker = MetricsTracker(save_path=os.path.join(cfg.log_dir, f"{exp_name}_metrics.json"))

    logger.info("══════════════════════════════════════")
    logger.info(f"🚀 Experiment: {exp_name}")
    logger.info(f"   Attention: {cfg.attn_type}")
    logger.info(f"   Method: {cfg.method}")
    logger.info(f"   Task: {cfg.task}")
    logger.info(f"   Model: {cfg.model_name}")
    logger.info("══════════════════════════════════════")

    set_all_seeds(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    # Model
    model = load_model(cfg, num_labels, device)
    model.to(device)

    # Trainer
    trainer = setup_trainer(model, tokenizer, dataset, cfg, tracker, logger)

    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    # Save final model and logs
    output_dir = os.path.join(cfg.output_dir, exp_name)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.json"))
    logger.info(f"Model and logs saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()