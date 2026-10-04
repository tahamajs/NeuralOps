#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning & Utility Functions
"""

import logging
import os
from pathlib import Path
from typing import Any

from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.utils import get_device, set_seed
from neuralops.data.dataset_loader import ExperimentConfig as data_ExperimentConfig
from neuralops.data.glue_loader import (
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
    load_glue_local,
)
from neuralops.data.preprocessing import preprocess_function as build_preprocess
from neuralops.trainer.callbacks import (
    MemoryCallback,
    SVDAttentionCallback,
    TimeCallback,
)
from neuralops.trainer.trainer_factory import get_trainer
from neuralops.utils.layer_norm_fix import fix_layernorm_naming

# ---------- Safe import of LoggingCallback ----------
try:
    from neuralops.trainer.callbacks import LoggingCallback as TrainerLoggingCallback
except ImportError:

    class TrainerLoggingCallback:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            """On log."""
            if logs:
                logging.getLogger(__name__).info(f"Step {state.global_step}: {logs}")


from neuralops.analysis.plotting import create_comprehensive_plots

logger = logging.getLogger(__name__)


class DjangoCallback(TrainerCallback):
    def __init__(self, progress_cb, log_cb, final_cb, stop_event=None) -> None:
        self.progress_cb = progress_cb
        self.log_cb = log_cb
        self.final_cb = final_cb
        self.stop_event = stop_event
        self._total_steps = None

    def on_train_begin(self, args, state, control, **kwargs) -> None:
        """On train begin."""
        self._total_steps = state.max_steps

    def on_log(self, args, state, control, logs=None, **kwargs) -> None:
        """On log."""
        if self.log_cb:
            self.log_cb(state.global_step, logs or {})

    def on_step_end(self, args, state, control, **kwargs) -> None:
        """On step end."""
        if self.stop_event and self.stop_event.is_set():
            control.should_training_stop = True
        if self.progress_cb and self._total_steps:
            pct = (state.global_step / self._total_steps) * 100
            self.progress_cb(state.global_step, self._total_steps, pct)

    def on_evaluate(self, args, state, control, metrics=None, **kwargs) -> None:
        """On evaluate."""
        if self.log_cb and metrics:
            self.log_cb(state.global_step, metrics)


def run_training(
    config,
    stop_event=None,
    progress_callback=None,
    log_callback=None,
    final_callback=None,
) -> Any:
    """Run training."""
    if isinstance(config, dict):
        cfg = data_ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    set_seed(cfg.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Data
    raw_datasets = load_glue_local(cfg.task, cfg.glue_base_path)
    num_labels, is_regression = get_num_labels_and_is_regression(cfg.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[cfg.task]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    preprocess_fn = build_preprocess(
        tokenizer, cfg.task, cfg.max_len, sent1_key, sent2_key, is_regression
    )
    dataset = raw_datasets.map(
        preprocess_fn, batched=True, remove_columns=raw_datasets["train"].column_names
    )
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )
    fix_layernorm_naming(model)
    model.to(device)

    # Trainer
    trainer_type = getattr(cfg, "trainer_type", "hf")
    trainer_cls = get_trainer(trainer_type)

    callbacks = [
        DjangoCallback(progress_callback, log_callback, final_callback, stop_event),
        TrainerLoggingCallback(None, logger),
        MemoryCallback(),
        TimeCallback(),
    ]
    if getattr(cfg, "compressor", None):
        callbacks.append(SVDAttentionCallback())

    trainer = trainer_cls.create(
        model=model,
        dataset=dataset,
        config=cfg,
        tokenizer=tokenizer,
        logger=logger,
        callbacks=callbacks,
    )

    logger.info("Starting training...")
    trainer.train()
    eval_results = trainer.evaluate()
    logger.info("Training completed.")

    # Save
    output_dir = Path(cfg.output_dir) / (cfg.exp_name or "training")
    output_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    try:
        create_comprehensive_plots(
            metrics=trainer.state.log_history,
            output_dir=str(output_dir / "plots"),
            prefix=cfg.exp_name or "training",
        )
    except Exception as e:
        logger.warning(f"Plot generation failed: {e}")

    if final_callback:
        final_callback(eval_results, trainer.state.log_history)

    return eval_results


def run_evaluation(config, model_path) -> Any:
    """Run evaluation."""
    if isinstance(config, dict):
        cfg = data_ExperimentConfig()
        for k, v in config.items():
            setattr(cfg, k, v)
    else:
        cfg = config

    set_seed(cfg.seed)
    device = get_device()

    # Reload data (simplified)
    raw_datasets = load_glue_local(cfg.task, cfg.glue_base_path)
    num_labels, is_regression = get_num_labels_and_is_regression(cfg.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[cfg.task]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    preprocess_fn = build_preprocess(
        tokenizer, cfg.task, cfg.max_len, sent1_key, sent2_key, is_regression
    )
    dataset = raw_datasets.map(preprocess_fn, batched=True)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["validation"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
    except Exception:
        logger.warning(f"Could not load from {model_path}, using base model")
        model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model_name, num_labels=num_labels
        )
    model.to(device)

    training_args = TrainingArguments(
        output_dir="./tmp_eval",
        per_device_eval_batch_size=cfg.batch_size * 2,
        remove_unused_columns=False,
        report_to="none",
    )
    eval_trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
    )
    eval_result = eval_trainer.evaluate()
    return {k.replace("eval_", ""): v for k, v in eval_result.items()}


def run_compression(config, output_dir) -> Any:
    """Run compression."""
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


def main() -> None:
    """Main."""
    from neuralops.experiment_logging.logger import LoggingConfig, MetricsTracker, setup_logging
    from neuralops.experiment_utils.config import build_config_from_args

    cfg = build_config_from_args()
    exp_name = cfg.exp_name or f"{cfg.attn_type}_{cfg.method}_{cfg.task}_{cfg.seed}"
    if not hasattr(cfg, "log_base_dir"):
        cfg.log_base_dir = cfg.log_dir
    if not hasattr(cfg, "use_timestamp"):
        cfg.use_timestamp = True
    if not hasattr(cfg, "run_id"):
        cfg.run_id = None

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

    raw_datasets = load_glue_local(cfg.task, cfg.glue_base_path)
    num_labels, is_regression = get_num_labels_and_is_regression(cfg.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[cfg.task]
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=True)

    preprocess_fn = build_preprocess(
        tokenizer, cfg.task, cfg.max_len, sent1_key, sent2_key, is_regression
    )
    dataset = raw_datasets.map(preprocess_fn, batched=True)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=num_labels
    )
    model.to(device)

    # Custom callback for tracker
    class TrackerCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs) -> None:
            """On log."""
            if logs:
                for k, v in logs.items():
                    tracker.log(k, v, step=state.global_step)

    trainer_cls = get_trainer(getattr(cfg, "trainer_type", "hf"))
    trainer = trainer_cls.create(
        model=model,
        dataset=dataset,
        config=cfg,
        tokenizer=tokenizer,
        logger=logger,
        callbacks=[TrackerCallback(), MemoryCallback(), TimeCallback()],
    )

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
