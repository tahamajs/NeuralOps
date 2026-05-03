#!/usr/bin/env python3
"""
Efficient Attention Framework – Interactive Training Launcher with Live Dashboard

Usage:
    python run.py                          # interactive config then train
    python run.py --config config.yaml     # non‑interactive from YAML
    python run.py --help
"""

import argparse, os, sys, time, threading, queue, yaml
import torch
import numpy as np
from pathlib import Path
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, DataCollatorWithPadding, set_seed,
    TrainerCallback
)

# Make framework importable
sys.path.insert(0, str(Path(__file__).parent))

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
from framework.analysis.metrics import MetricsTracker
from experiment_logging import setup_logging, LoggingCallback
from framework.config import CompressionConfig

# UI libraries
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False
    print("⚠️  Install questionary for interactive mode: pip install questionary")

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.live import Live
    from rich.layout import Layout
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("⚠️  Install rich for better output: pip install rich")

import matplotlib
matplotlib.use('TkAgg')  # or 'QtAgg' – opens separate window
import matplotlib.pyplot as plt

# Global plot queue for live updates
plot_queue = queue.Queue()
training_active = threading.Event()
current_metrics = {}

# ================================================================
# Live plotting callback
# ================================================================
class LivePlotCallback(TrainerCallback):
    def __init__(self):
        self.train_loss = []
        self.eval_loss = []
        self.train_steps = []
        self.eval_steps = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs:
            if "loss" in logs and "eval_loss" not in logs:
                self.train_loss.append(logs["loss"])
                self.train_steps.append(state.global_step)
            if "eval_loss" in logs:
                self.eval_loss.append(logs["eval_loss"])
                self.eval_steps.append(state.global_step)
            plot_queue.put(("update", {
                "train_steps": self.train_steps,
                "train_loss": self.train_loss,
                "eval_steps": self.eval_steps,
                "eval_loss": self.eval_loss,
                "global_step": state.global_step,
            }))

    def on_train_end(self, args, state, control, **kwargs):
        plot_queue.put(("done", None))


def live_plotter():
    """Starts a separate thread that updates a matplotlib figure from plot_queue."""
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.canvas.manager.set_window_title("Training Dashboard")
    lines1, = ax1.plot([], [], 'b-', label="Train Loss")
    lines2, = ax2.plot([], [], 'r-o', label="Eval Loss")
    ax1.set_title("Training Loss")
    ax2.set_title("Evaluation Loss")
    ax1.set_xlabel("Step")
    ax2.set_xlabel("Step")
    ax1.legend()
    ax2.legend()
    ax1.grid(True, alpha=0.3)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=False)

    while training_active.is_set() or not plot_queue.empty():
        try:
            cmd, data = plot_queue.get(timeout=0.5)
            if cmd == "update" and data:
                lines1.set_data(data["train_steps"], data["train_loss"])
                lines2.set_data(data["eval_steps"], data["eval_loss"])
                ax1.relim(); ax1.autoscale_view()
                ax2.relim(); ax2.autoscale_view()
                fig.canvas.draw()
                fig.canvas.flush_events()
            elif cmd == "done":
                break
        except queue.Empty:
            pass
        time.sleep(0.1)
    plt.ioff()
    plt.close(fig)


# ================================================================
# Interactive config builder
# ================================================================
def interactive_config():
    """Ask user for all training parameters via questionary."""
    if not HAS_QUESTIONARY:
        print("questionary not installed – using defaults.")
        return default_config()

    # Model
    model_choices = [
        questionary.Choice("distilbert-base-uncased", value="distilbert-base-uncased"),
        questionary.Choice("bert-base-uncased", value="bert-base-uncased"),
        questionary.Choice("roberta-base", value="roberta-base"),
        questionary.Choice("custom (enter name)", value="custom"),
    ]
    model = questionary.select("Choose base model:", choices=model_choices).ask()
    if model == "custom":
        model = questionary.text("Enter HuggingFace model name:").ask()

    # Task
    task_choices = ["cola", "sst2", "mrpc", "qqp", "stsb", "mnli", "qnli", "rte", "wnli"]
    task = questionary.select("Choose GLUE task:", choices=task_choices).ask()

    # Attention
    attn_choices = list(ATTENTION_REGISTRY.keys()) if ATTENTION_REGISTRY else ["full", "aurora", "nova"]
    attention = questionary.select("Choose attention type:", choices=attn_choices).ask()

    # Training method
    method_choices = ["baseline", "lora", "rdrop", "r3f", "fisher_lora", "fisher_gelora", "llrd"]
    method = questionary.select("Training method:", choices=method_choices).ask()

    # Hyperparams
    epochs = int(questionary.text("Number of epochs:", default="5").ask())
    batch_size = int(questionary.text("Batch size:", default="32").ask())
    lr = float(questionary.text("Learning rate:", default="1e-5").ask())
    max_len = int(questionary.text("Max sequence length:", default="128").ask())

    # Compression rank (if low‑rank attention)
    rank = int(questionary.text("Compression rank:", default="64").ask())

    # Additional compressor
    use_compressor = questionary.confirm("Apply additional compression?", default=False).ask()
    compressor = None
    if use_compressor:
        comp_choices = list(COMPRESSOR_REGISTRY.keys()) if COMPRESSOR_REGISTRY else ["svd", "tt"]
        compressor = questionary.select("Compressor:", choices=comp_choices).ask()

    # Output
    output_dir = questionary.text("Output directory:", default="./experiments").ask()

    return {
        "model": model,
        "task": task,
        "attention": attention,
        "method": method,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "max_len": max_len,
        "rank": rank,
        "compressor": compressor,
        "output_dir": output_dir,
    }

def default_config():
    return {
        "model": "distilbert-base-uncased",
        "task": "sst2",
        "attention": "full",
        "method": "baseline",
        "epochs": 5,
        "batch_size": 32,
        "lr": 1e-5,
        "max_len": 128,
        "rank": 64,
        "compressor": None,
        "output_dir": "./experiments",
    }

def load_yaml_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_compression_config_from_args(args, yaml_cfg=None):
    """Convert args (or yaml) into CompressionConfig."""
    base = {}
    if yaml_cfg:
        base.update(yaml_cfg)
    base.update(vars(args))
    return CompressionConfig(
        attn_type=args.attn_type,
        dropout=getattr(args, 'dropout', 0.1),
        lora_r=getattr(args, 'lora_r', 8),
        lora_alpha=getattr(args, 'lora_alpha', 16),
        lora_dropout=getattr(args, 'lora_dropout', 0.1),
        rdrop_alpha=getattr(args, 'rdrop_alpha', 5.0),
        r3f_epsilon=getattr(args, 'r3f_epsilon', 1e-5),
        llrd_decay=getattr(args, 'llrd_decay', 0.95),
        mixout_p=getattr(args, 'mixout_p', 0.9),
        dislora_beta=getattr(args, 'dislora_beta', 0.5),
        dislora_capacity_factor=getattr(args, 'dislora_capacity_factor', 1.0),
        fisher_num_batches=getattr(args, 'fisher_num_batches', 20),
        fisher_sample_size=getattr(args, 'fisher_sample_size', 200),
        gelora_margin=getattr(args, 'gelora_margin', 1.0),
        gelora_min_rank=getattr(args, 'gelora_min_rank', 2),
        gelora_max_rank=getattr(args, 'gelora_max_rank', 32),
        fisher_scale_max=getattr(args, 'fisher_scale_max', 2.0),
        use_lion=getattr(args, 'use_lion', False),
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        early_stopping_patience=getattr(args, 'early_stopping_patience', None),
        max_grad_norm=getattr(args, 'max_grad_norm', 1.0),
        num_landmarks=getattr(args, 'num_landmarks', 16),
        feature_dim=getattr(args, 'feature_dim', 128),
        k_early=getattr(args, 'k_early', 32),
        k_mid=getattr(args, 'k_mid', 16),
        k_late=getattr(args, 'k_late', 12),
        use_gate=not getattr(args, 'no_gate', False),
        progressive=True,
        max_seq_len=args.max_len,
    )

# ================================================================
# Main training logic
# ================================================================
def run_training(args, cfg_dict):
    """Execute training with given configuration (interactive or YAML)."""
    console = Console() if HAS_RICH else None
    device = get_device()
    set_all_seeds(args.seed)

    if console:
        console.rule("[bold green]Starting Experiment")

    # Logging
    exp_name = f"{cfg_dict['attention']}_{cfg_dict['method']}_{cfg_dict['task']}_{args.seed}"
    logger = setup_logging(log_dir="./logs", experiment_name=exp_name)
    tracker = MetricsTracker(os.path.join("./logs", f"{exp_name}_metrics.json"))

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Model: {cfg_dict['model']}, Task: {cfg_dict['task']}, "
                f"Attention: {cfg_dict['attention']}, Method: {cfg_dict['method']}")

    # Tokenizer & Dataset
    tokenizer = AutoTokenizer.from_pretrained(cfg_dict['model'], use_fast=True)
    raw_datasets = load_glue_local(cfg_dict['task'], "./datasets/glue")
    num_labels, is_regression = get_num_labels_and_is_regression(cfg_dict['task'])
    sent1_key, sent2_key = GLUE_TASK_KEYS[cfg_dict['task']]

    pre_fn = build_preprocess(tokenizer, cfg_dict['task'], cfg_dict['max_len'],
                             sent1_key, sent2_key, is_regression)
    dataset = raw_datasets.map(pre_fn, batched=True,
                               remove_columns=raw_datasets["train"].column_names)
    cols = ["input_ids", "attention_mask", "labels"]
    if "token_type_ids" in dataset["train"].column_names:
        cols.append("token_type_ids")
    dataset.set_format(type="torch", columns=cols)

    # Load model
    config = AutoConfig.from_pretrained(cfg_dict['model'], num_labels=num_labels)
    model = AutoModelForSequenceClassification.from_pretrained(cfg_dict['model'], config=config)
    fix_layernorm_naming(model)

    # Build CompressionConfig
    comp_config = build_compression_config_from_args(args)

    # Inject attention
    if cfg_dict['attention'] != "full":
        inject_fn = ATTENTION_REGISTRY.get(cfg_dict['attention'])
        if inject_fn is None:
            raise ValueError(f"Unknown attention: {cfg_dict['attention']}")
        if hasattr(model, "distilbert"):
            layers = model.distilbert.transformer.layer
        elif hasattr(model, "bert"):
            layers = model.bert.encoder.layer
        else:
            raise ValueError("Unsupported model architecture")
        for i, layer in enumerate(layers):
            old_attn = layer.attention
            inject_fn(layer, config, comp_config, i, len(layers), old_attn)
        logger.info(f"Injected '{cfg_dict['attention']}' into {len(layers)} layers.")

    # Additional compression
    if cfg_dict.get('compressor'):
        compressor_cls = COMPRESSOR_REGISTRY.get(cfg_dict['compressor'])
        if compressor_cls:
            compressor = compressor_cls(model, rank=cfg_dict['rank'])
            model, _ = compressor.compress()
            logger.info(f"Applied additional compression: {cfg_dict['compressor']}")

    model.to(device)
    if not hasattr(model, 'get_ortho_loss'):
        model.get_ortho_loss = lambda: torch.tensor(0.0, device=device)

    # Apply training method (simplified – if you need full method integration, activate METHOD_REGISTRY)
    # (For brevity, we only handle baseline/lora here; extend as needed)
    if cfg_dict['method'] not in ("baseline",):
        logger.warning(f"Method '{cfg_dict['method']}' not fully integrated in this demo; using baseline.")

    # Metric
    metric_map = {"cola": "eval_matthews_correlation", "stsb": "eval_pearson",
                  "mrpc": "eval_f1", "qqp": "eval_f1"}
    metric_for_best = metric_map.get(cfg_dict['task'], "eval_accuracy")

    # Training args
    output_dir = os.path.join(cfg_dict['output_dir'], exp_name)
    os.makedirs(output_dir, exist_ok=True)
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=cfg_dict['epochs'],
        per_device_train_batch_size=cfg_dict['batch_size'],
        per_device_eval_batch_size=cfg_dict['batch_size'] * 2,
        learning_rate=cfg_dict['lr'],
        weight_decay=0.01,
        lr_scheduler_type="linear",
        warmup_steps=500,
        logging_steps=50,
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
        compute_metrics=get_compute_metrics(cfg_dict['task']),
    )

    # Live plot callback
    live_cb = LivePlotCallback()
    trainer.add_callback(live_cb)
    trainer.add_callback(LoggingCallback(tracker, logger))

    # Start live plotter thread
    training_active.set()
    plot_thread = threading.Thread(target=live_plotter, daemon=True)
    plot_thread.start()

    logger.info("Training started – watch the dashboard!")
    trainer.train()

    # Cleanup
    training_active.clear()
    plot_thread.join(timeout=2)

    eval_result = trainer.evaluate()
    logger.info("=" * 60)
    logger.info("FINAL EVALUATION:")
    for k, v in eval_result.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info("=" * 60)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    tracker.save(os.path.join(output_dir, "metrics.json"))
    logger.info(f"Model saved to {output_dir}")
    if console:
        console.rule("[bold green]Experiment Complete")

# ================================================================
# CLI entry
# ================================================================
def main():
    parser = argparse.ArgumentParser(description="Efficient Attention GLUE Runner with Live Dashboard")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML configuration file")
    parser.add_argument("--interactive", action="store_true", default=None,
                        help="Force interactive mode even without questionary")
    parser.add_argument("--seed", type=int, default=42)
    # These are used when not interactive
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--attn_type", type=str, default=None)
    parser.add_argument("--method", type=str, default="baseline")
    parser.add_argument("--task", "-t", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max_len", type=int, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--compressor", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="./experiments")
    args = parser.parse_args()

    # Determine config source
    if args.config:
        yaml_cfg = load_yaml_config(args.config)
        cfg_dict = {**default_config(), **yaml_cfg}
    elif args.interactive or (HAS_QUESTIONARY and args.model is None):
        cfg_dict = interactive_config()
    else:
        # Use CLI args with defaults
        cfg_dict = default_config()
        if args.model: cfg_dict['model'] = args.model
        if args.attn_type: cfg_dict['attention'] = args.attn_type
        if args.task: cfg_dict['task'] = args.task
        if args.epochs: cfg_dict['epochs'] = args.epochs
        if args.batch_size: cfg_dict['batch_size'] = args.batch_size
        if args.lr: cfg_dict['lr'] = args.lr
        if args.max_len: cfg_dict['max_len'] = args.max_len
        if args.rank: cfg_dict['rank'] = args.rank
        if args.compressor: cfg_dict['compressor'] = args.compressor
        cfg_dict['output_dir'] = args.output_dir

    # Ensure all necessary fields are present
    required = ['model', 'task', 'attention', 'method']
    for key in required:
        if key not in cfg_dict:
            cfg_dict[key] = default_config()[key]

    run_training(args, cfg_dict)


if __name__ == "__main__":
    main()