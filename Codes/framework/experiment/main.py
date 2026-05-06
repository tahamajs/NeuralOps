#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning & Utility Functions
Now uses:
  - framework.core.utils       (set_seed, get_device)
  - framework.core.registry    (ATTENTION_REGISTRY, COMPRESSOR_REGISTRY)
  - framework.data             (task_mapper, glue_loader, preprocessing)
  - framework.trainer          (get_trainer, callbacks, results manager)
  - framework.analysis         (MetricsTracker, plotting, report generation)
  - framework.utils            (layer_norm_fix, checkpoint, save_results)
  - framework.compression_engine.core.registry (for compression)
"""
import sys
import os
import json
import logging
import torch
import numpy as np
from pathlib import Path
from transformers import Trainer, TrainerCallback, AutoTokenizer, AutoConfig, AutoModelForSequenceClassification

# Core utilities (seed, device)
from framework.core.utils import set_seed, get_device, count_parameters, format_number

# Registry (attention, compressor, trainer, visualizer)
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, TRAINER_REGISTRY

# Data loading & task mapping
from framework.data.dataset_loader import load_dataset, ExperimentConfig as data_ExperimentConfig
from framework.data.task_mapper import get_metric_name, get_num_labels, get_text_columns
from framework.data.glue_loader import load_glue_local, GLUE_TASK_KEYS, get_num_labels_and_is_regression
from framework.data.preprocessing import preprocess_function as build_preprocess

# Model loading & layer normalization fix
from framework.utils.layer_norm_fix import fix_layernorm_naming

# Trainer (factory + callbacks)
from framework.trainer.trainer_factory import get_trainer
from framework.trainer.callbacks import (
    MemoryCallback, TimeCallback, GradientNormCallback,
    SVDAttentionCallback, LoggingCallback as TrainerLoggingCallback
)
from framework.trainer.results_manager import ResultsManager

# Analysis (metrics tracking, plotting)
from framework.analysis.plotting import create_comprehensive_plots
from framework.analysis.metrics import MetricsTracker as AnalysisMetricsTracker

# Additional utils
from framework.utils.checkpoint import CheckpointManager
from framework.utils.save_results import save_experiment_results

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


# ----------------------------------------------------------------------
# Training entry point (used by Django runner)
# ----------------------------------------------------------------------
def run_training(config, stop_event=None,
                 progress_callback=None, log_callback=None, final_callback=None):
    """
    Full training pipeline leveraging the entire framework.
    config: dict or ExperimentConfig instance.
    Returns evaluation metrics dict after training.
    """
    # Normalise config
    if isinstance(config, dict):
        cfg = data_ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    # ---- Setup ----
    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # ---- Data Loading ----
    # Use framework.data.glue_loader for GLUE tasks
    raw_datasets = load_glue_local(cfg.task, cfg.glue_base_path)
    num_labels, is_regression = get_num_labels_and_is_regression(cfg.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[cfg.task]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    preprocess_fn = build_preprocess(tokenizer, cfg.task, cfg.max_len,
                                     sent1_key, sent2_key, is_regression)
    dataset = raw_datasets.map(preprocess_fn, batched=True,
                               remove_columns=raw_datasets["train"].column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    # ---- Model Loading ----
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels)
    fix_layernorm_naming(model)
    model.to(device)

    # ---- Trainer Setup ----
    # Use the trainer factory (supports hf, unsloth, rl, custom_loop)
    trainer_cls = get_trainer(cfg.trainer_type if hasattr(cfg, 'trainer_type') else 'hf')
    trainer = trainer_cls.create(
        model=model,
        dataset=dataset,
        config=cfg,
        tokenizer=tokenizer,
        logger=logger,
        callbacks=[
            DjangoCallback(progress_callback, log_callback, final_callback, stop_event),
            TrainerLoggingCallback(None, logger),  # basic logging
            MemoryCallback(),
            TimeCallback(),
            SVDAttentionCallback() if cfg.compressor else None,
        ]
    )

    # ---- Training ----
    logger.info("Starting training...")
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info("Training completed.")

    # ---- Save outputs ----
    output_dir = Path(cfg.output_dir) / (cfg.exp_name or "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Generate comprehensive plots using framework.analysis
    try:
        create_comprehensive_plots(
            metrics=trainer.state.log_history,
            output_dir=str(output_dir / "plots"),
            prefix=cfg.exp_name or "training"
        )
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    if final_callback:
        final_callback(eval_results, trainer.state.log_history)

    return eval_results


# ----------------------------------------------------------------------
# Evaluation runner (standalone)
# ----------------------------------------------------------------------
def run_evaluation(config, model_path):
    if isinstance(config, dict):
        cfg = data_ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    set_seed(cfg.seed)
    device = get_device()

    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception:
        logger.warning(f"Could not load model from {model_path}, falling back to base model")
        model = load_model(cfg, num_labels, device)

    model.to(device)

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
    cleaned = {k.replace("eval_", ""): v for k, v in eval_result.items()}
    return cleaned


# ----------------------------------------------------------------------
# Compression runner (using framework.compression_engine)
# ----------------------------------------------------------------------
def run_compression(config, output_dir):
    if isinstance(config, dict):
        cfg = data_ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    compressor_name = cfg.compressor or "svd"
    if compressor_name not in COMPRESSOR_REGISTRY:
        raise ValueError(f"Unknown compressor: {compressor_name}")

    set_seed(cfg.seed)
    device = get_device()

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name)
    model.to(device)

    compressor_cls = COMPRESSOR_REGISTRY[compressor_name]
    compressor = compressor_cls(model, rank=cfg.compression_rank)
    compressed_model = compressor.compress()

    os.makedirs(output_dir, exist_ok=True)
    compressed_model.save_pretrained(output_dir)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.save_pretrained(output_dir)

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
# CLI entry point (unchanged)
# ----------------------------------------------------------------------
def main():
    from framework.experiment.config import build_config_from_args
    cfg = build_config_from_args()

    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"
    if not hasattr(cfg, 'log_base_dir'):
        cfg.log_base_dir = cfg.log_dir
    if not hasattr(cfg, 'use_timestamp'):
        cfg.use_timestamp = True
    if not hasattr(cfg, 'run_id'):
        cfg.run_id = None

    from experiment_logging.logger import setup_logging, MetricsTracker, LoggingConfig
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

    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    tokenizer, dataset, num_labels, is_regression = load_and_preprocess_data(cfg)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    model = AutoModelForSequenceClassification.from_pretrained(cfg.model_name, num_labels=num_labels)
    model.to(device)

    trainer = setup_trainer(model, tokenizer, dataset, cfg, tracker, logger)

    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()

    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    output_dir = os.path.join(cfg.output_dir, exp_name)
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.json"))
    logger.info(f"Model and logs saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()