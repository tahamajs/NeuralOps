#!/usr/bin/env python
"""
╔══════════════════════════════════════════════════════════════╗
║     EFFICIENT ATTENTION FRAMEWORK – GLUE FINE‑TUNING        ║
╚══════════════════════════════════════════════════════════════╝
Usage:
    python main.py --attn_type leka_v2 --task sst2 --epochs 5
    python main.py --attn_type spectra_v2 --model bert-base-uncased --task mrpc
"""

import argparse, os, torch, logging
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, set_seed
)
from datasets import DatasetDict

from framework.data.glue_loader import (
    load_glue_local, GLUE_TASK_KEYS, get_num_labels_and_is_regression
)
from framework.data.preprocessing import preprocess_function as build_preprocess
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY
from framework.trainer.trainer import CompressionTrainer, get_compute_metrics
from framework.utils.device import get_device
from framework.utils.seed import set_all_seeds
from framework.utils.layer_norm_fix import fix_layernorm_naming
from framework.config import CompressionConfig
from logging import setup_logging, MetricsTracker

# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Efficient Attention GLUE Runner")
    # Model & Data
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--attn_type", type=str, required=True, help="Registered attention type (e.g., leka_v2, spectra_v2, full)")
    parser.add_argument("--compressor", type=str, default=None, help="Registered compressor name")
    parser.add_argument("--task", "-t", type=str, default="sst2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32, help="Per device")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--log_dir", type=str, default="./logs")
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name")
    # Additional attention config (optional)
    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_gate", action="store_true")
    args = parser.parse_args()

    # Setup logging
    exp_name = args.exp_name or f"{args.attn_type}_{args.task}_{args.seed}"
    logger = setup_logging(log_dir=args.log_dir, experiment_name=exp_name)
    tracker = MetricsTracker(save_path=os.path.join(args.log_dir, f"{exp_name}_metrics.json"))

    logger.info("══════════════════════════════════════")
    logger.info(f"🚀 Experiment: {exp_name}")
    logger.info(f"   Attention: {args.attn_type}")
    logger.info(f"   Compressor: {args.compressor if args.compressor else 'none'}")
    logger.info(f"   Task: {args.task}")
    logger.info(f"   Model: {args.model}")
    logger.info("══════════════════════════════════════")

    set_all_seeds(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # Load tokenizer & dataset
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw_datasets = load_glue_local(args.task, "/Users/tahamajs/Documents/uni/Project/datasets/glue")
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    # Preprocess
    pre_fn = build_preprocess(tokenizer, args.task, args.max_len, sent1_key, sent2_key, is_regression)
    dataset = raw_datasets.map(pre_fn, batched=True, remove_columns=raw_datasets["train"].column_names)

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)
    logger.info(f"Train samples: {len(dataset['train'])}, Validation: {len(dataset.get('validation', []))}")

    # Load model
    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    fix_layernorm_naming(model)

    # Build compression config
    comp_config = CompressionConfig(
        attn_type=args.attn_type,
        dropout=0.1,
        num_landmarks=args.num_landmarks,
        feature_dim=args.feature_dim,
        k_early=args.k_early,
        k_mid=args.k_mid,
        k_late=args.k_late,
        use_gate=not args.no_gate,
        progressive=True,
        max_seq_len=args.max_len,
    )

    # Inject attention
    if args.attn_type != "full":
        inject_fn = ATTENTION_REGISTRY.get(args.attn_type)
        if inject_fn is None:
            raise ValueError(f"Unknown attention type: {args.attn_type}. Available: {ATTENTION_REGISTRY.keys()}")
        # Determine architecture
        if hasattr(model, "distilbert"):
            layers = model.distilbert.transformer.layer
        elif hasattr(model, "bert"):
            layers = model.bert.encoder.layer
        else:
            raise ValueError("Unsupported model architecture")
        for i, layer in enumerate(layers):
            old_attn = layer.attention
            inject_fn(layer, config, comp_config, i, len(layers), old_attn)
        logger.info(f"Injected '{args.attn_type}' into {len(layers)} layers.")
    else:
        logger.info("Using full attention (no injection).")

    model.to(device)
    # Ensure get_ortho_loss exists
    if not hasattr(model, 'get_ortho_loss'):
        def dummy_ortho_loss():
            return torch.tensor(0.0, device=device)
        model.get_ortho_loss = dummy_ortho_loss

    # Apply additional compression if requested
    if args.compressor:
        comp_class = COMPRESSOR_REGISTRY.get(args.compressor)
        if comp_class:
            compressor = comp_class(model, energy_threshold=0.95)
            model, stats = compressor.compress()
            logger.info(f"Compression applied: {stats}")
        else:
            logger.warning(f"Compressor '{args.compressor}' not found. Available: {COMPRESSOR_REGISTRY.keys()}")

    # Metric name
    metric_map = {"cola": "eval_matthews_correlation", "stsb": "eval_pearson",
                  "mrpc": "eval_f1", "qqp": "eval_f1"}
    metric_for_best = metric_map.get(args.task, "eval_accuracy")

    # Training arguments
    output_dir = os.path.join(args.output_dir, exp_name)
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    # Trainer
    trainer = CompressionTrainer(
        ortho_weight=0.0,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(args.task),
    )

    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    # Save everything
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.json"))
    logger.info(f"Model and logs saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()
