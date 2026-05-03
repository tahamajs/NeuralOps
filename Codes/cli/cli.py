#!/usr/bin/env python3
"""
Efficient Transformer Toolkit – CLI
====================================
Unified command‑line interface for training, evaluation, compression, and data generation.

Usage:
    python cli.py train --model distilbert-base-uncased --task mrpc --attention aurora
    python cli.py evaluate --model_path ./output --task mrpc
    python cli.py compress --model distilbert-base-uncased --method low_rank --output compressed_model
    python cli.py generate-synthetic --output data/synthetic.csv
"""

import os
import sys
import json
import click
import torch
from pathlib import Path
from typing import Optional

# Add parent directory to path so framework is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from framework.data.dataset_loader import ExperimentConfig, load_dataset
from framework.data.preprocessing import preprocess_function
from framework.data.task_mapper import get_num_labels, get_metric_name
from framework.models.model_loader import load_model
from framework.attentions.core.registry import ATTENTION_REGISTRY
from framework.compression.core.registry import COMPRESSOR_REGISTRY

# ---------------------------------------------------------------------------
# Common options
# ---------------------------------------------------------------------------
def experiment_config_from_args(**kwargs):
    """Build ExperimentConfig from CLI arguments."""
    return ExperimentConfig(**kwargs)

def common_model_options(func):
    """Decorator for model‑related options."""
    options = [
        click.option('--model', '-m', default='distilbert-base-uncased', help='HF model name or path'),
        click.option('--attention', '-a', default='full', help='Attention type (registered key)'),
        click.option('--rank', '-r', default=64, help='Compression rank'),
        click.option('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device'),
        click.option('--use-unsloth', is_flag=True, help='Use Unsloth for loading'),
        click.option('--cache-dir', default=None, help='Cache directory for HF models'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func

def common_data_options(func):
    """Decorator for dataset options."""
    options = [
        click.option('--task', '-t', default='mrpc', help='Task name (GLUE or custom file)'),
        click.option('--max-seq-length', default=128, help='Tokenization max length'),
        click.option('--split-ratio', default=0.1, help='Validation split for custom datasets'),
        click.option('--glue-path', default='./data/glue', help='Local GLUE data path'),
    ]
    for opt in reversed(options):
        func = opt(func)
    return func

# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------
@click.group()
def cli():
    """Efficient Transformer Toolkit – command line tools."""
    pass

# ---------------------------------------------------------------------------
# train
# ---------------------------------------------------------------------------
@cli.command()
@common_model_options
@common_data_options
@click.option('--output', '-o', default='./output', help='Output directory')
@click.option('--epochs', default=3, help='Number of training epochs')
@click.option('--batch-size', default=16, help='Training batch size')
@click.option('--lr', default=2e-5, help='Learning rate')
@click.option('--num-labels', default=None, type=int, help='Number of labels (auto‑detected if not set)')
def train(model, attention, rank, device, use_unsloth, cache_dir,
          task, max_seq_length, split_ratio, glue_path,
          output, epochs, batch_size, lr, num_labels):
    """Train a model with a specific attention mechanism on a dataset."""
    click.echo(f"🚀 Training {model} with attention '{attention}' on task '{task}'")

    # 1. Load model with attention
    click.echo("Loading model...")
    model, tokenizer = load_model(
        model_name=model,
        num_labels=num_labels or 2,  # will be overridden later if GLUE
        attention_type=attention,
        rank=rank,
        device=device,
        use_unsloth=use_unsloth,
        cache_dir=cache_dir
    )

    # 2. Build experiment config for data
    data_cfg = ExperimentConfig(
        dataset_name=task,
        max_seq_length=max_seq_length,
        split_ratio=split_ratio,
        glue_base_path=glue_path,
        num_labels=num_labels or 2
    )
    dataset_dict = load_dataset(data_cfg)

    # Auto‑detect num_labels from GLUE mapping if not provided
    if task in {"cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"}:
        from framework.data.glue_loader import get_num_labels_and_is_regression
        n_labels, is_reg = get_num_labels_and_is_regression(task)
        if num_labels is None:
            data_cfg.num_labels = n_labels
    else:
        if num_labels is None:
            # try to infer from dataset
            if "label" in dataset_dict["train"].column_names:
                data_cfg.num_labels = len(dataset_dict["train"].unique("label"))

    # Update model's classifier if needed
    if hasattr(model, 'config') and data_cfg.num_labels != model.config.num_labels:
        model.config.num_labels = data_cfg.num_labels
        if hasattr(model, 'num_labels'):
            model.num_labels = data_cfg.num_labels
        # Reinitialise the classifier head? For simplicity we skip reinit.

    # 3. Tokenize dataset
    click.echo("Tokenizing dataset...")
    from framework.data.preprocessing import preprocess_function
    sent1_key, sent2_key = None, None
    if task in {"mrpc","qqp","stsb","mnli","qnli","rte","wnli"}:
        cols = {"mrpc":("sentence1","sentence2"), "qqp":("question1","question2"),
                "stsb":("sentence1","sentence2"), "mnli":("premise","hypothesis"),
                "qnli":("question","sentence"), "rte":("sentence1","sentence2"),
                "wnli":("sentence1","sentence2")}
        sent1_key, sent2_key = cols[task]
    elif task in {"cola","sst2"}:
        sent1_key = "sentence"
    else:
        # custom dataset: try to auto‑detect
        cols = dataset_dict["train"].column_names
        if len(cols) >= 2:
            sent1_key = cols[0]
            if len(cols) >= 3:
                sent2_key = cols[1]

    preprocess_fn = preprocess_function(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        sent1_key=sent1_key,
        sent2_key=sent2_key,
        is_regression=(task=="stsb")
    )
    encoded = dataset_dict.map(preprocess_fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    # 4. Train
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=output,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{output}/logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation", encoded.get("test")),
    )

    click.echo("Training...")
    trainer.train()

    # Save final model
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    click.echo(f"✅ Model saved to {output}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
@cli.command()
@click.option('--model_path', '-mp', required=True, help='Path to trained model')
@click.option('--task', '-t', default='mrpc', help='Task name')
@click.option('--attention', '-a', default='full', help='Attention type (if needed)')
@click.option('--rank', '-r', default=64)
@click.option('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu')
@click.option('--glue-path', default='./data/glue')
@click.option('--max-seq-length', default=128)
@click.option('--batch-size', default=16)
def evaluate(model_path, task, attention, rank, device, glue_path, max_seq_length, batch_size):
    """Evaluate a trained model on a GLUE task."""
    click.echo(f"📊 Evaluating {model_path} on {task}")

    # Load model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    # Load dataset
    data_cfg = ExperimentConfig(dataset_name=task, glue_base_path=glue_path, max_seq_length=max_seq_length)
    dataset_dict = load_dataset(data_cfg)

    # Tokenize
    from framework.data.preprocessing import preprocess_function
    sent1_key, sent2_key = None, None
    if task in {"mrpc","qqp","stsb","mnli","qnli","rte","wnli"}:
        cols = {"mrpc":("sentence1","sentence2"), "qqp":("question1","question2"),
                "stsb":("sentence1","sentence2"), "mnli":("premise","hypothesis"),
                "qnli":("question","sentence"), "rte":("sentence1","sentence2"),
                "wnli":("sentence1","sentence2")}
        sent1_key, sent2_key = cols[task]
    elif task in {"cola","sst2"}:
        sent1_key = "sentence"

    fn = preprocess_function(tokenizer, max_seq_length=max_seq_length,
                             sent1_key=sent1_key, sent2_key=sent2_key,
                             is_regression=(task=="stsb"))
    encoded = dataset_dict.map(fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(
        output_dir="/tmp/eval",
        per_device_eval_batch_size=batch_size,
    )
    trainer = Trainer(model=model, args=args,
                      eval_dataset=encoded.get("validation", encoded.get("test")))
    results = trainer.evaluate()
    click.echo(results)


# ---------------------------------------------------------------------------
# compress
# ---------------------------------------------------------------------------
@cli.command()
@click.option('--model', '-m', default='distilbert-base-uncased', help='Model name or path')
@click.option('--method', default='low_rank', help='Compression method (low_rank, pruning, quant, hybrid)')
@click.option('--target-sparsity', default=0.5, help='Sparsity for pruning')
@click.option('--target-rank', default=64, help='Rank for low‑rank methods')
@click.option('--output', '-o', default='./compressed_model', help='Output directory')
@click.option('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
def compress(model, method, target_sparsity, target_rank, output, device):
    """Compress a model using a chosen compression method."""
    click.echo(f"🔧 Compressing {model} with method '{method}'")

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model)

    # Get compressor from registry
    compressor_cls = COMPRESSOR_REGISTRY.get(method)
    if compressor_cls is None:
        click.echo(f"Unknown method '{method}'. Available: {list(COMPRESSOR_REGISTRY.keys())}")
        return

    compressor = compressor_cls(
        model=base_model,
        target_sparsity=target_sparsity,
        target_rank=target_rank,
        method=method,
    )
    compressed_model, stats = compressor.compress()

    click.echo(f"Compression stats: {stats}")
    compressed_model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    click.echo(f"✅ Compressed model saved to {output}")


# ---------------------------------------------------------------------------
# generate‑synthetic
# ---------------------------------------------------------------------------
@cli.command()
@click.option('--output', '-o', default='data/synthetic.csv', help='Output CSV path')
@click.option('--samples', default=5000, help='Number of total samples')
@click.option('--domain', default='movies', help='Domain (movies, tech, restaurants, books)')
@click.option('--seed', default=42)
def generate_synthetic(output, samples, domain, seed):
    """Generate a synthetic classification dataset."""
    from framework.data.synthetic_generator import generate_synthetic_data
    generate_synthetic_data(output_path=output, num_samples=samples, domain=domain, seed=seed)


# ---------------------------------------------------------------------------
# list‑attentions
# ---------------------------------------------------------------------------
@cli.command()
def list_attentions():
    """Show all registered attention types."""
    click.echo("Available attention types:")
    for key in ATTENTION_REGISTRY.keys():
        click.echo(f"  - {key}")


if __name__ == "__main__":
    cli()