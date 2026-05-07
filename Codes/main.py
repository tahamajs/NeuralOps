#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning
Usage:
    python main.py --attn_type nexus --compressor svd --method lora --task sst2
    python main.py --attn_type spectra_v2 --compressor none --method rdrop --task mrpc
"""

import argparse
import os
import torch
import yaml
import importlib   # added for lazy importing
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)

os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------- Framework imports ----------
from framework.data.glue_loader import (
    load_glue_local,
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
)
from framework.data.preprocessing import preprocess_function as build_preprocess
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from framework.trainer.trainer import CompressionTrainer, get_compute_metrics
# Remove duplicate: from framework.trainer.methods.registry import METHOD_REGISTRY
from framework.trainer.methods.rdrop_r3f import MethodTrainer
from framework.utils.device import get_device
from framework.utils.seed import set_all_seeds
from framework.utils.layer_norm_fix import fix_layernorm_naming
from framework.config import CompressionConfig
from framework.experiment_logging.config import LoggingConfig
from framework.experiment_logging.logger import setup_logging
from framework.experiment_logging.tracker import MetricsTracker
from framework.experiment_logging.callbacks import LoggingCallback


# ---------- YAML helpers ----------
def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def build_compression_config(yaml_cfg, args):
    """Build CompressionConfig from YAML + CLI overrides."""
    cfg_dict = yaml_cfg.copy() if yaml_cfg else {}
    # CLI overrides (only if provided)
    cli_map = {
        'attn_type': args.attn_type,
        'dropout': getattr(args, 'dropout', None),
        'lora_r': args.lora_r,
        'lora_alpha': args.lora_alpha,
        'lora_dropout': args.lora_dropout,
        'rdrop_alpha': args.rdrop_alpha,
        'r3f_epsilon': args.r3f_epsilon,
        'llrd_decay': args.llrd_decay,
        'mixout_p': args.mixout_p,
        'dislora_beta': args.dislora_beta,
        'dislora_capacity_factor': args.dislora_capacity_factor,
        'fisher_num_batches': args.fisher_num_batches,
        'fisher_sample_size': args.fisher_sample_size,
        'gelora_margin': args.gelora_margin,
        'gelora_min_rank': args.gelora_min_rank,
        'gelora_max_rank': args.gelora_max_rank,
        'fisher_scale_max': args.fisher_scale_max,
        'use_lion': args.use_lion,
        'gradient_checkpointing': args.gradient_checkpointing,
        'early_stopping_patience': args.early_stopping_patience,
        'max_grad_norm': args.max_grad_norm,
        'num_landmarks': args.num_landmarks,
        'feature_dim': args.feature_dim,
        'k_early': args.k_early,
        'k_mid': args.k_mid,
        'k_late': args.k_late,
        'use_gate': not args.no_gate,
        'progressive': True,
        'max_seq_len': args.max_len,
    }
    for key, val in cli_map.items():
        if val is not None:
            cfg_dict[key] = val
    # Filter to only valid CompressionConfig fields
    valid_keys = {f.name for f in CompressionConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return CompressionConfig(**filtered)


# -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Efficient Attention GLUE Runner")
    # Model & Data
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--attn_type", type=str, required=True,
                        help=f"Attention type.")
    parser.add_argument("--compressor", type=str, default=None,
                        help=f"Compressor to apply before training.")
    parser.add_argument("--method", type=str, default="baseline",
                        help=f"Training method. Available: baseline plus those registered.")
    parser.add_argument("--task", "-t", type=str, default="sst2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default=None)

    # Method‑specific parameters (unchanged)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--rdrop_alpha", type=float, default=5.0)
    parser.add_argument("--r3f_epsilon", type=float, default=1e-5)
    parser.add_argument("--mixout_p", type=float, default=0.9)
    parser.add_argument("--llrd_decay", type=float, default=0.95)
    parser.add_argument("--dislora_beta", type=float, default=0.5)
    parser.add_argument("--dislora_capacity_factor", type=float, default=1.0)
    parser.add_argument("--fisher_num_batches", type=int, default=20)
    parser.add_argument("--fisher_sample_size", type=int, default=200)
    parser.add_argument("--gelora_margin", type=float, default=1.0)
    parser.add_argument("--gelora_min_rank", type=int, default=2)
    parser.add_argument("--gelora_max_rank", type=int, default=32)
    parser.add_argument("--fisher_scale_max", type=float, default=2.0)
    parser.add_argument("--use_lion", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--early_stopping_patience", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Attention‑specific params
    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_gate", action="store_true")

    args = parser.parse_args()

    # ---------- Setup experiment directory and logging ----------
    exp_name = args.exp_name or f"{args.attn_type}_{args.method}_{args.task}_{args.seed}"
    output_dir = os.path.join(args.output_dir, exp_name)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    log_cfg = LoggingConfig(
        log_level="INFO",
        log_format="structured",
        log_to_console=True,
        log_to_file=True,
        log_dir=log_dir,
        experiment_name=exp_name,
        run_id=str(args.seed),
        use_timestamp=True,
        metrics_enabled=True,
    )
    logger = setup_logging(log_cfg)
    tracker = MetricsTracker(save_path=os.path.join(log_dir, "metrics.jsonl"))

    logger.info("══════════════════════════════════════")
    logger.info(f"🚀 Experiment: {exp_name}")
    logger.info(f"   Attention: {args.attn_type}")
    logger.info(f"   Compressor: {args.compressor or 'none'}")
    logger.info(f"   Method: {args.method}")
    logger.info(f"   Task: {args.task}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Output dir: {output_dir}")
    logger.info("══════════════════════════════════════")

    set_all_seeds(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

    # ---------- Load data ----------
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    raw_datasets = load_glue_local(args.task, "/Users/tahamajs/Documents/uni/Project/datasets/glue")
    num_labels, is_regression = get_num_labels_and_is_regression(args.task)
    sent1_key, sent2_key = GLUE_TASK_KEYS[args.task]

    pre_fn = build_preprocess(tokenizer, args.task, args.max_len, sent1_key, sent2_key, is_regression)
    dataset = raw_datasets.map(pre_fn, batched=True, remove_columns=raw_datasets["train"].column_names)

    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)
    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")

    # ---------- Load model ----------
    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    fix_layernorm_naming(model)

    # ---------- Build compression config ----------
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        comp_config = build_compression_config(yaml_cfg, args)
    else:
        comp_config = CompressionConfig(
            attn_type=args.attn_type,
            dropout=0.1,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            rdrop_alpha=args.rdrop_alpha,
            r3f_epsilon=args.r3f_epsilon,
            llrd_decay=args.llrd_decay,
            mixout_p=args.mixout_p,
            dislora_beta=args.dislora_beta,
            dislora_capacity_factor=args.dislora_capacity_factor,
            fisher_num_batches=args.fisher_num_batches,
            fisher_sample_size=args.fisher_sample_size,
            gelora_margin=args.gelora_margin,
            gelora_min_rank=args.gelora_min_rank,
            gelora_max_rank=args.gelora_max_rank,
            fisher_scale_max=args.fisher_scale_max,
            use_lion=args.use_lion,
            gradient_checkpointing=args.gradient_checkpointing,
            early_stopping_patience=args.early_stopping_patience,
            max_grad_norm=args.max_grad_norm,
            num_landmarks=args.num_landmarks,
            feature_dim=args.feature_dim,
            k_early=args.k_early,
            k_mid=args.k_mid,
            k_late=args.k_late,
            use_gate=not args.no_gate,
            progressive=True,
            max_seq_len=args.max_len,
        )

    # ---------- 1. Inject attention (with lazy import) ----------
    if args.attn_type != "full":
        # Dynamically import the attention module
        try:
            module_path = f"framework.attentions.{args.attn_type}"
            importlib.import_module(module_path)
            logger.debug(f"Lazy-loaded attention module: {module_path}")
        except ImportError as e:
            raise ImportError(f"Could not import attention module for '{args.attn_type}': {e}")

        inject_fn = ATTENTION_REGISTRY.get(args.attn_type)
        if inject_fn is None:
            raise ValueError(f"Attention type '{args.attn_type}' not registered after import. Available: {list(ATTENTION_REGISTRY.keys())}")

        # Locate transformer layers (DistilBERT or BERT)
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

    # ---------- 2. Apply compressor (with lazy import) ----------
    if args.compressor and args.compressor != "none":
        # Dynamically import the compressor module
        try:
            module_path = f"framework.compression_engine.methods.{args.compressor}"
            importlib.import_module(module_path)
            logger.debug(f"Lazy-loaded compressor module: {module_path}")
        except ImportError as e:
            raise ImportError(f"Could not import compressor module for '{args.compressor}': {e}")

        compress_fn = COMPRESSOR_REGISTRY.get(args.compressor)
        if compress_fn is None:
            raise ValueError(f"Unknown compressor: {args.compressor}. Available: {list(COMPRESSOR_REGISTRY.keys())}")
        model = compress_fn(model, comp_config)
        logger.info(f"Applied compressor: {args.compressor}")

    model.to(device)

    # Dummy ortho loss for trainers that require it
    if not hasattr(model, 'get_ortho_loss'):
        def dummy_ortho_loss():
            return torch.tensor(0.0, device=device)
        model.get_ortho_loss = dummy_ortho_loss

    # ---------- 3. Apply training method (optional lazy import) ----------
    if args.method != "baseline":
        if args.method not in METHOD_REGISTRY:
            # Optionally lazy-import method modules if they are not auto-loaded
            try:
                module_path = f"framework.trainer.methods.{args.method}"
                importlib.import_module(module_path)
            except ImportError:
                pass  # method might already be registered elsewhere
        if args.method not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method: {args.method}. Available: {list(METHOD_REGISTRY.keys())}")
        apply_fn = METHOD_REGISTRY[args.method]
        if args.method in ("fisher_lora", "fisher_gelora"):
            apply_fn(model, comp_config, dataset["train"], tokenizer, GLUE_TASK_KEYS[args.task])
        else:
            apply_fn(model, comp_config)
        logger.info(f"Applied method: {args.method}")

    # ---------- Training arguments ----------
    metric_map = {
        "cola": "eval_matthews_correlation",
        "stsb": "eval_pearson",
        "mrpc": "eval_f1",
        "qqp": "eval_f1",
    }
    metric_for_best = metric_map.get(args.task, "eval_accuracy")

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

    # ---------- Trainer ----------
    if args.method in ("rdrop", "r3f"):
        trainer_cls = MethodTrainer
        extra = dict(
            method=args.method,
            rdrop_alpha=args.rdrop_alpha,
            r3f_epsilon=args.r3f_epsilon,
            is_regression=is_regression
        )
    else:
        trainer_cls = CompressionTrainer
        extra = {}

    trainer = trainer_cls(
        ortho_weight=0.0,
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation", dataset.get("validation_matched")),
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=get_compute_metrics(args.task),
        **extra
    )

    trainer.add_callback(LoggingCallback(tracker, logger))

    # ---------- Train and evaluate ----------
    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    # ---------- Save artifacts ----------
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.jsonl"))
    logger.info(f"Model and logs saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")


if __name__ == "__main__":
    main()