#!/usr/bin/env python3
"""
Single experiment runner for the Efficient Transformer Toolkit.
Usage:
    python experiments/run_experiment.py --model distilbert-base-uncased \
        --attention aurora --rank 64 --task mrpc --output results/my_run
"""

import argparse
import sys
from pathlib import Path
import torch

# Make sure the framework is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression_engine.core.registry import COMPRESSOR_REGISTRY
from framework.models.model_loader import load_model
from framework.data.dataset_loader import load_dataset, ExperimentConfig
from framework.data.preprocessing import preprocess_function
from framework.training import get_trainer, ResultsManager, get_task_metrics
from framework.analysis import LoggingCallback, MetricsTracker
from framework.visualization import SVDAttentionVisualizer
from framework.utils.device import get_device
from framework.utils.helpers import set_seed, count_parameters, format_time
from framework.utils.logger import setup_logger

import logging
import time

def parse_args():
    parser = argparse.ArgumentParser(description="Run a single compression experiment")
    # Model
    parser.add_argument("--model", "-m", default="distilbert-base-uncased", help="HF model name")
    parser.add_argument("--attention", "-a", default="full", help="Attention type (registry key)")
    parser.add_argument("--rank", "-r", type=int, default=64, help="Compression rank")
    parser.add_argument("--device", "-d", default=None, help="Device (cpu/cuda)")
    parser.add_argument("--use-unsloth", action="store_true", help="Use Unsloth for loading")
    parser.add_argument("--cache-dir", default=None, help="HF cache directory")
    # Data
    parser.add_argument("--task", "-t", default="mrpc", help="GLUE task or custom CSV/JSON")
    parser.add_argument("--max-seq-length", type=int, default=128, help="Tokenization max length")
    parser.add_argument("--split-ratio", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--glue-path", default="./data/glue", help="Local GLUE data path")
    # Training
    parser.add_argument("--output", "-o", default=None, help="Base output directory (default: auto-named in results/)")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fp16", action="store_true", default=torch.cuda.is_available(), help="Use mixed precision")
    # Compression
    parser.add_argument("--compressor", default=None, help="Additional compression method (registry key) applied after attention")
    parser.add_argument("--compressor-rank", type=int, default=None, help="Rank for additional compressor")
    # Reporting
    parser.add_argument("--no-plots", action="store_true", help="Skip generating plots")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    set_seed(args.seed)

    # Device
    device = get_device(args.device or "auto")

    # ------------------------------------------------------------
    # 1. Results manager (creates unique run directory)
    # ------------------------------------------------------------
    results = ResultsManager(
        experiment_name=args.task + "_" + args.attention,
        base_dir=args.output or "results",
        config=vars(args),
        log_to_file=True
    )
    logger = results.logger

    logger.info(f"Starting experiment: {args.task} with {args.attention} (rank={args.rank})")
    logger.info(f"Run directory: {results.run_dir}")

    # ------------------------------------------------------------
    # 2. Load model with attention injection
    # ------------------------------------------------------------
    logger.info("Loading model...")
    model, tokenizer = load_model(
        model_name=args.model,
        num_labels=None,  # will be set by dataset later
        attention_type=args.attention,
        rank=args.rank,
        device=device,
        use_unsloth=args.use_unsloth,
        cache_dir=args.cache_dir
    )

    # ------------------------------------------------------------
    # 3. Load and tokenize dataset
    # ------------------------------------------------------------
    logger.info("Loading dataset...")
    data_cfg = ExperimentConfig(
        dataset_name=args.task,
        dataset_path=args.task if args.task.endswith((".csv", ".json", ".parquet")) else None,
        max_seq_length=args.max_seq_length,
        split_ratio=args.split_ratio,
        glue_base_path=args.glue_path,
        num_labels=None
    )
    dataset_dict = load_dataset(data_cfg)

    # Determine task type and columns
    from framework.data.glue_loader import get_num_labels_and_is_regression
    if args.task in {"cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"}:
        num_labels, is_regression = get_num_labels_and_is_regression(args.task)
        data_cfg.num_labels = num_labels
    else:
        if "label" in dataset_dict["train"].column_names:
            num_labels = len(set(dataset_dict["train"]["label"]))
            is_regression = False
        else:
            num_labels = 2
            is_regression = False
    data_cfg.num_labels = num_labels

    # Update model's number of labels if needed
    if hasattr(model, 'config') and model.config.num_labels != num_labels:
        model.config.num_labels = num_labels
        if hasattr(model, 'num_labels'):
            model.num_labels = num_labels

    # Tokenization
    pair_tasks = {
        "mrpc": ("sentence1", "sentence2"),
        "qqp": ("question1", "question2"),
        "stsb": ("sentence1", "sentence2"),
        "mnli": ("premise", "hypothesis"),
        "qnli": ("question", "sentence"),
        "rte": ("sentence1", "sentence2"),
        "wnli": ("sentence1", "sentence2")
    }
    single_tasks = {"cola": "sentence", "sst2": "sentence"}

    if args.task in pair_tasks:
        sent1_key, sent2_key = pair_tasks[args.task]
    elif args.task in single_tasks:
        sent1_key, sent2_key = single_tasks[args.task], None
    else:
        # Auto-detect
        cols = dataset_dict["train"].column_names
        sent1_key = cols[0]
        sent2_key = cols[1] if len(cols) > 2 else None

    preprocess_fn = preprocess_function(
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        sent1_key=sent1_key,
        sent2_key=sent2_key,
        is_regression=(args.task == "stsb")
    )
    encoded = dataset_dict.map(preprocess_fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # ------------------------------------------------------------
    # 4. (Optional) Additional compression
    # ------------------------------------------------------------
    if args.compressor:
        logger.info(f"Applying additional compression: {args.compressor}")
        compressor_cls = COMPRESSOR_REGISTRY.get(args.compressor)
        if compressor_cls:
            compressor = compressor_cls(model, rank=args.compressor_rank or 64)
            model, _ = compressor.compress()
            logger.info(f"Parameters after additional compression: {count_parameters(model):,}")

    # ------------------------------------------------------------
    # 5. Training
    # ------------------------------------------------------------
    logger.info("Starting training...")
    compute_metrics_fn = get_task_metrics(args.task.lower())

    training_config = {
        "experiment_name": args.task + "_" + args.attention,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_grad_norm": 1.0,
        "eval_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "fp16": args.fp16,
        "logging_steps": 50,
        "logging_dir": str(results.log_dir),
    }

    trainer, _ = get_trainer(
        model=model,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation") or encoded.get("test"),
        results_manager=results,
        config=training_config,
        compute_metrics=compute_metrics_fn
    )

    start_time = time.time()
    trainer.train()
    training_time = time.time() - start_time
    logger.info(f"Training completed in {format_time(training_time)}")

    # ------------------------------------------------------------
    # 6. Final evaluation and reporting
    # ------------------------------------------------------------
    logger.info("Running final evaluation...")
    final_metrics = trainer.evaluate()
    results.log_metrics({"final_eval": final_metrics, "training_time_s": training_time})
    logger.info(f"Final metrics: {final_metrics}")

    # Generate visualizations
    if not args.no_plots:
        logger.info("Generating visualizations...")
        viz = SVDAttentionVisualizer(output_dir=str(results.run_dir / "plots"))
        # You could add custom plots here, e.g.:
        # viz.plot_rank_comparison(...)
        results.save_plots()

    # Save final model
    results.save_final_model(model, tokenizer)

    logger.info("Experiment complete.")

if __name__ == "__main__":
    main()