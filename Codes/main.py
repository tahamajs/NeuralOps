#!/usr/bin/env python
"""
Efficient Attention Framework – GLUE Fine‑Tuning with full logging and analysis.
"""

import framework.attentions
import framework.compression_engine

import argparse
import os
import sys
import torch
import yaml
import json
import time
import importlib
import transformers
from datetime import datetime
from pathlib import Path
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    TrainingArguments,
    DataCollatorWithPadding,
    set_seed,
)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ["HF_ENDPOINT"] = "https://hf.devneeds.ir"
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "models")
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from framework.data.glue_loader import (
    load_glue_local,
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
)
from framework.data.preprocessing import preprocess_function as build_preprocess
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY
from framework.trainer.trainer import CompressionTrainer, get_compute_metrics
from framework.trainer.methods.rdrop_r3f import MethodTrainer
from framework.utils.device import get_device
from framework.utils.seed import set_all_seeds
from framework.utils.layer_norm_fix import fix_layernorm_naming
from framework.config import CompressionConfig
from framework.experiment_logging.config import LoggingConfig
from framework.experiment_logging.logger import setup_logging
from framework.experiment_logging.tracker import MetricsTracker
from framework.experiment_logging.callbacks import LoggingCallback

# Optional callbacks (graceful fallback)
try:
    from framework.callbacks.external import WandbCallback, TensorBoardCallback, WebSocketCallback
except ImportError:
    WandbCallback = TensorBoardCallback = WebSocketCallback = None
try:
    from framework.callbacks.custom import (
        EMACallback, SWACallback, GradientClippingCallback,
        FreezeLayersCallback, UnfreezeScheduleCallback, NanCheckerCallback, ActivationCheckpointingCallback
    )
except ImportError:
    EMACallback = SWACallback = GradientClippingCallback = None
    FreezeLayersCallback = UnfreezeScheduleCallback = NanCheckerCallback = ActivationCheckpointingCallback = None
try:
    from framework.callbacks.analysis import (
        SVDAttentionCallback, AttentionHeatmapCallback,
        GradientNormCallback, LayerwiseAnalysisCallback, ActivationStatsCallback
    )
except ImportError:
    SVDAttentionCallback = AttentionHeatmapCallback = GradientNormCallback = None
    LayerwiseAnalysisCallback = ActivationStatsCallback = None

def import_compressor_module(compressor_name):
    possible_paths = [
        f"framework.compression_engine.methods.{compressor_name}",
        f"framework.compression_engine.methods.{compressor_name}_compressor",
        f"framework.compression_engine.{compressor_name}",
        f"framework.compression_engine.{compressor_name}_compressor",
    ]
    for path in possible_paths:
        try:
            importlib.import_module(path)
            print(f"✅ Loaded compressor from: {path}")
            return True
        except ModuleNotFoundError:
            continue
    raise ImportError(f"Could not find compressor module for '{compressor_name}'")

def load_yaml_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def build_compression_config(yaml_cfg, args):
    cfg_dict = yaml_cfg.copy() if yaml_cfg else {}
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
    valid_keys = {f.name for f in CompressionConfig.__dataclass_fields__.values()}
    filtered = {k: v for k, v in cfg_dict.items() if k in valid_keys}
    return CompressionConfig(**filtered)

def save_attention_maps(model, dataloader, device, output_dir, num_samples=1, max_heads=4):
    model.eval()
    attn_dir = Path(output_dir) / "attention_maps"
    attn_dir.mkdir(exist_ok=True, parents=True)
    batch = next(iter(dataloader))
    batch = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad():
        outputs = model(**batch, output_attentions=True)
    attentions = outputs.attentions
    for layer_idx, attn_tensor in enumerate(attentions):
        attn = attn_tensor[0].cpu().numpy()
        n_heads = min(attn.shape[0], max_heads)
        fig, axes = plt.subplots(1, n_heads, figsize=(4*n_heads, 4))
        if n_heads == 1:
            axes = [axes]
        for h in range(n_heads):
            sns.heatmap(attn[h], ax=axes[h], cmap='viridis', cbar=True)
            axes[h].set_title(f'Layer {layer_idx}, Head {h}')
            axes[h].set_xlabel('Key')
            axes[h].set_ylabel('Query')
        plt.tight_layout()
        plt.savefig(attn_dir / f'layer{layer_idx}_heads.png')
        plt.close()
    model.train()
    return str(attn_dir)

def plot_training_curves(history, output_dir):
    df = pd.DataFrame(history)
    if df.empty:
        return
    train_df = df[df['loss'].notna()] if 'loss' in df else pd.DataFrame()
    eval_df = df[df['eval_loss'].notna()] if 'eval_loss' in df else pd.DataFrame()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    if not train_df.empty:
        axes[0].plot(train_df['step'], train_df['loss'], label='Train loss')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training loss')
        axes[0].legend()
    if not eval_df.empty:
        axes[1].plot(eval_df['epoch'], eval_df['eval_loss'], label='Val loss', marker='o')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Validation loss')
        axes[1].legend()
    plt.tight_layout()
    plt.savefig(Path(output_dir) / 'training_curves.png')
    plt.close()

def save_dataset_metadata(dataset, output_dir, task_name, num_labels, is_regression):
    meta = {
        "task": task_name,
        "num_labels": num_labels,
        "is_regression": is_regression,
        "train_size": len(dataset["train"]),
        "val_size": len(dataset.get("validation", [])),
        "column_names": dataset["train"].column_names,
    }
    with open(Path(output_dir) / "dataset_metadata.json", 'w') as f:
        json.dump(meta, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default="distilbert-base-uncased")
    parser.add_argument("--attn_type", type=str, required=True)
    parser.add_argument("--compressor", type=str, default=None)
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--task", "-t", type=str, default="sst2")
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    parser.add_argument("--exp_name", type=str, default=None)

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
    parser.add_argument("--max_train_samples", type=int, default=None)

    parser.add_argument("--num_landmarks", type=int, default=16)
    parser.add_argument("--feature_dim", type=int, default=128)
    parser.add_argument("--k_early", type=int, default=32)
    parser.add_argument("--k_mid", type=int, default=16)
    parser.add_argument("--k_late", type=int, default=12)
    parser.add_argument("--no_gate", action="store_true")
    parser.add_argument("--num_train_epochs", type=int, default=None)

    parser.add_argument("--save_visuals", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--use_swa", action="store_true")
    parser.add_argument("--gradient_clip_norm", type=float, default=None)
    parser.add_argument("--svd_analysis_steps", type=int, default=0)
    parser.add_argument("--attention_heatmap_steps", type=int, default=0)
    parser.add_argument("--log_grad_norm", action="store_true")
    parser.add_argument("--activation_stats", action="store_true")

    args = parser.parse_args()

    epochs = args.num_train_epochs if args.num_train_epochs is not None else args.epochs
    start_time = time.time()
    start_time_iso = datetime.now().isoformat()

    exp_name = args.exp_name or f"{args.attn_type}_{args.method}_{args.task}_{args.seed}"
    output_dir = Path(args.output_dir) / exp_name
    log_dir = output_dir / "logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(exist_ok=True)

    log_cfg = LoggingConfig(
        log_level="INFO",
        log_format="structured",
        log_to_console=True,
        log_to_file=True,
        log_dir=str(log_dir),
        experiment_name=exp_name,
        run_id=str(args.seed),
        use_timestamp=True,
        metrics_enabled=True,
    )
    logger = setup_logging(log_cfg)
    tracker = MetricsTracker(save_path=str(log_dir / "metrics.jsonl"))

    logger.info("══════════════════════════════════════")
    logger.info(f"🚀 Experiment: {exp_name}")
    logger.info(f"   Attention: {args.attn_type}")
    logger.info(f"   Compressor: {args.compressor or 'none'}")
    logger.info(f"   Method: {args.method}")
    logger.info(f"   Task: {args.task}")
    logger.info(f"   Model: {args.model}")
    logger.info(f"   Output dir: {output_dir}")
    logger.info(f"   Start time: {start_time_iso}")
    logger.info("══════════════════════════════════════")

    set_all_seeds(args.seed)
    device = get_device()
    logger.info(f"Using device: {device}")

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

    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
        logger.info(f"Limited training to {len(dataset['train'])} samples")

    logger.info(f"Train: {len(dataset['train'])}, Val: {len(dataset.get('validation', []))}")
    save_dataset_metadata(dataset, output_dir, args.task, num_labels, is_regression)

    config = AutoConfig.from_pretrained(args.model, num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, config=config)
    fix_layernorm_naming(model)

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

# ---------- 1. Inject attention ----------
    if args.attn_type != "full":
        inj = ATTENTION_REGISTRY.get(args.attn_type)
        if inj is None:
            raise ValueError(f"Attention type '{args.attn_type}' not registered. Available: {list(ATTENTION_REGISTRY.keys())}")

        # Create a unified injector that works for both classes and functions
        import inspect


        def make_injector(obj):
            # If the object is a class, we need to instantiate it.
            if inspect.isclass(obj):
                sig = inspect.signature(obj.__init__)
                # Count number of parameters excluding 'self'
                params = list(sig.parameters.values())[1:]  # skip self
                n_required = sum(1 for p in params if p.default == inspect.Parameter.empty and p.kind != p.VAR_KEYWORD)
                n_total = len(params)
                
                def inject(layer, config, comp_config, layer_idx, num_layers, old_attn):
                    # Prepare a dict of possible arguments
                    kwargs = {}
                    # Try to pass as many as possible based on parameter names
                    arg_map = {
                        'config': config,
                        'comp_config': comp_config,
                        'layer_idx': layer_idx,
                        'num_layers': num_layers,
                        'old_attn': old_attn,
                        'attention': old_attn,  # some might expect 'attention'
                        'model_config': config,
                        'layer_id': layer_idx,
                        'rank': getattr(comp_config, 'rank', None),
                        'hidden_size': getattr(config, 'hidden_size', None),
                        'num_heads': getattr(config, 'num_attention_heads', None),
                    }
                    for p in params:
                        if p.name in arg_map:
                            kwargs[p.name] = arg_map[p.name]
                        elif p.kind == p.VAR_KEYWORD:
                            # **kwargs case - pass everything
                            kwargs.update(arg_map)
                            break
                    # If still missing required arguments, fallback to positional
                    try:
                        new_attn = obj(**kwargs)
                    except TypeError:
                        # Fallback: call with the first n_required positional args
                        positional_args = []
                        for p in params[:n_required]:
                            if p.name in arg_map:
                                positional_args.append(arg_map[p.name])
                            else:
                                # default to None or a placeholder
                                positional_args.append(None)
                        new_attn = obj(*positional_args)
                    layer.attention = new_attn
                return inject
            else:
                # For a function, check its signature and call with appropriate args.
                sig = inspect.signature(obj)
                params = list(sig.parameters.values())
                
                def inject(layer, config, comp_config, layer_idx, num_layers, old_attn):
                    kwargs = {}
                    arg_map = {
                        'config': config, 'comp_config': comp_config, 'layer_idx': layer_idx,
                        'num_layers': num_layers, 'old_attn': old_attn, 'attention': old_attn,
                        'model_config': config, 'layer_id': layer_idx, 'rank': getattr(comp_config, 'rank', None),
                        'hidden_size': getattr(config, 'hidden_size', None), 'num_heads': getattr(config, 'num_attention_heads', None),
                    }
                    for p in params:
                        if p.name in arg_map:
                            kwargs[p.name] = arg_map[p.name]
                        elif p.kind == inspect.Parameter.VAR_KEYWORD:
                            kwargs.update(arg_map)
                            break
                    # Also provide 'layer' and maybe 'layer_attention' if needed? but typical injection functions already take layer.
                    # If the function expects the layer as first argument, we should pass it.
                    # We can detect if the first parameter is named 'layer' and its type may be the module.
                    if params and params[0].name == 'layer':
                        # The registered function likely expects (layer, config, ... ) as per the original pattern.
                        # In that case, we call it with the original arguments directly.
                        try:
                            obj(layer, config, comp_config, layer_idx, num_layers, old_attn)
                        except:
                            # fallback to kwargs
                            obj(**kwargs)
                    else:
                        obj(**kwargs)
                return inject
            



        inject_fn = make_injector(inj)

        # Locate transformer layers (DistilBERT or BERT)
        if hasattr(model, "distilbert"):
            layers = model.distilbert.transformer.layer
        elif hasattr(model, "bert"):
            layers = model.bert.encoder.layer
        else:
            raise ValueError("Unsupported model architecture")

        for i, layer in enumerate(layers):
            old_attn = layer.attention
            try:
                inject_fn(layer, config, comp_config, i, len(layers), old_attn)
            except Exception as e:
                logger.error(f"Failed to inject attention '{args.attn_type}' in layer {i}: {e}")
                raise

        logger.info(f"Injected '{args.attn_type}' into {len(layers)} layers.")
    else:
        logger.info("Using full attention (no injection).")



    # Compressor
    if args.compressor and args.compressor != "none":
        compress_fn = COMPRESSOR_REGISTRY.get(args.compressor)
        if compress_fn is None:
            raise ValueError(f"Unknown compressor: {args.compressor}. Available: {list(COMPRESSOR_REGISTRY.keys())}")
        model = compress_fn(model, comp_config)
        logger.info(f"Applied compressor: {args.compressor}")



    model.to(device)
    if not hasattr(model, 'get_ortho_loss'):
        def dummy_ortho_loss():
            return torch.tensor(0.0, device=device)
        model.get_ortho_loss = dummy_ortho_loss

    # Method
    if args.method != "baseline":
        if args.method not in METHOD_REGISTRY:
            try:
                importlib.import_module(f"framework.trainer.methods.{args.method}")
            except ImportError:
                pass
        if args.method not in METHOD_REGISTRY:
            raise ValueError(f"Unknown method: {args.method}")
        apply_fn = METHOD_REGISTRY[args.method]
        if args.method in ("fisher_lora", "fisher_gelora"):
            apply_fn(model, comp_config, dataset["train"], tokenizer, GLUE_TASK_KEYS[args.task])
        else:
            apply_fn(model, comp_config)
        logger.info(f"Applied method: {args.method}")

    metric_map = {"cola": "eval_matthews_correlation", "stsb": "eval_pearson", "mrpc": "eval_f1", "qqp": "eval_f1"}
    metric_for_best = metric_map.get(args.task, "eval_accuracy")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "checkpoints"),
        num_train_epochs=epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=5,
        logging_steps=2,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best,
        greater_is_better=True,
        fp16=False,
        remove_unused_columns=False,
        report_to="none",
    )

    if args.method in ("rdrop", "r3f"):
        trainer_cls = MethodTrainer
        extra = dict(method=args.method, rdrop_alpha=args.rdrop_alpha, r3f_epsilon=args.r3f_epsilon, is_regression=is_regression)
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

    # Optional monitoring callbacks
    if args.use_wandb and WandbCallback is not None:
        trainer.add_callback(WandbCallback(project="transformer-toolkit"))
    if args.use_tensorboard and TensorBoardCallback is not None:
        trainer.add_callback(TensorBoardCallback(log_dir=str(output_dir / "tensorboard")))
    if args.use_ema and EMACallback is not None:
        trainer.add_callback(EMACallback(decay=0.999))
    if args.use_swa and SWACallback is not None:
        trainer.add_callback(SWACallback(start_epoch=max(1, epochs//2), swa_lr=args.lr*0.1))
    if args.gradient_clip_norm is not None and GradientClippingCallback is not None:
        trainer.add_callback(GradientClippingCallback(max_norm=args.gradient_clip_norm))
    if NanCheckerCallback is not None:
        trainer.add_callback(NanCheckerCallback())
    if args.gradient_checkpointing and ActivationCheckpointingCallback is not None:
        trainer.add_callback(ActivationCheckpointingCallback(modules_to_checkpoint=["distilbert.transformer.layer"]))
    if args.svd_analysis_steps > 0 and SVDAttentionCallback is not None:
        trainer.add_callback(SVDAttentionCallback(log_every_steps=args.svd_analysis_steps))
    if args.attention_heatmap_steps > 0 and AttentionHeatmapCallback is not None:
        trainer.add_callback(AttentionHeatmapCallback(output_dir=str(output_dir), log_every_steps=args.attention_heatmap_steps))
    if args.log_grad_norm and GradientNormCallback is not None:
        trainer.add_callback(GradientNormCallback())
    if args.activation_stats and ActivationStatsCallback is not None:
        trainer.add_callback(ActivationStatsCallback())

    logger.info("Starting training...")
    trainer.train()
    eval_result = trainer.evaluate()
    logger.info("="*60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("="*60)

    # Save full config dump
    full_config = {
        "command_line": vars(args),
        "model_config": config.to_dict(),
        "compression_config": comp_config.__dict__,
        "training_args": training_args.to_dict(),
        "datasets": {
            "train_size": len(dataset["train"]),
            "val_size": len(dataset.get("validation", [])),
            "num_labels": num_labels,
            "is_regression": is_regression,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "timestamp_start": start_time_iso,
    }
    with open(output_dir / "full_config.json", "w") as f:
        json.dump(full_config, f, indent=2, default=str)

    # Save per-step metrics
    with open(output_dir / "per_step_metrics.jsonl", "w") as f:
        for log in trainer.state.log_history:
            f.write(json.dumps(log, default=str) + "\n")

    # Plot and save training curves
    history = trainer.state.log_history
    plot_training_curves(history, output_dir)
    if history:
        pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
        logger.info("Saved training history to training_history.csv")

    if args.save_visuals:
        eval_loader = trainer.get_eval_dataloader()
        attn_dir = save_attention_maps(model, eval_loader, device, output_dir)
        logger.info(f"Attention maps saved to {attn_dir}")

    end_time = time.time()
    final_metrics = {
        "start_time": datetime.fromtimestamp(start_time).isoformat(),
        "end_time": datetime.fromtimestamp(end_time).isoformat(),
        "duration_seconds": end_time - start_time,
        "final_accuracy": eval_result.get("eval_accuracy", 0.0),
        "final_loss": eval_result.get("eval_loss", 0.0),
        "best_metric": eval_result.get(metric_for_best, 0.0),
        "num_parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "model_name": args.model,
        "attention_type": args.attn_type,
        "compressor": args.compressor,
        "method": args.method,
        "task": args.task,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "epochs": epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
    }
    with open(output_dir / "final_metrics.json", 'w') as f:
        json.dump(final_metrics, f, indent=2)
    logger.info(f"✅ All results saved to {output_dir}")
    logger.info("🎉 Experiment completed successfully!")

if __name__ == "__main__":
    main()