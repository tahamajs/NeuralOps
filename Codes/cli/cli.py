#!/usr/bin/env python3
"""
Efficient Transformer Toolkit – CLI
====================================

Command‑line interface with:
  • YAML config files (pre‑built profiles)
  • Config precedence: defaults → YAML → env vars → CLI
  • Interactive mode with questionary
  • Rich terminal output
  • All registered attention types & GLUE benchmarks

Usage:
  python cli.py train --preset fast
  python cli.py train --config my_experiment.yaml
  python cli.py train --interactive
  python cli.py train --task mrpc --attention aurora --model bert_base
  python cli.py list-configs
  python cli.py train --print-config
"""

import sys
import shutil
from pathlib import Path
from typing import Optional

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.syntax import Syntax
from rich import print as rprint

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from config_loader import (
    TrainingConfig,
    load_config,
    list_available_configs,
    resolve_config_sources,
    CONFIGS_DIR,
)

console = Console()

# ────────────────────────────────────────────────────────────────
# Helpers
# ────────────────────────────────────────────────────────────────

def _build_config(
    task: Optional[str] = None,
    model: Optional[str] = None,
    attention: Optional[str] = None,
    preset: Optional[str] = None,
    config_files: Optional[tuple] = None,
    overrides: Optional[dict] = None,
) -> TrainingConfig:
    """Build final config from all sources."""
    paths = resolve_config_sources(
        task=task, model=model, attention=attention, preset=preset,
        custom_configs=[Path(p) for p in config_files] if config_files else None
    )
    return load_config(config_paths=paths, cli_overrides=overrides)


def _print_config(cfg: TrainingConfig):
    """Pretty‑print a config using rich."""
    yaml_str = yaml.dump(
        {k: v for k, v in cfg.__dict__.items() if not k.startswith("_")},
        default_flow_style=False, sort_keys=False
    )
    console.print(Panel(Syntax(yaml_str, "yaml", theme="monokai"), title="Resolved Configuration"))


def _show_available():
    """Display all available config profiles."""
    configs = list_available_configs()
    for category, names in configs.items():
        table = Table(title=f"[bold]{category.upper()}[/bold]")
        table.add_column("Name", style="cyan")
        for name in names:
            table.add_row(name)
        console.print(table)

    # Also show available attention types
    try:
        from framework.attentions.core.registry import ATTENTION_REGISTRY
        att_table = Table(title="[bold]REGISTERED ATTENTIONS[/bold]")
        att_table.add_column("Key", style="green")
        att_table.add_column("Count")
        keys = list(ATTENTION_REGISTRY.keys())
        att_table.add_row(", ".join(keys), str(len(keys)))
        console.print(att_table)
    except ImportError:
        pass


def _interactive_config() -> TrainingConfig:
    """Interactive config builder using questionary (if available)."""
    try:
        import questionary
    except ImportError:
        console.print("[yellow]questionary not installed. Install with: pip install questionary[/yellow]")
        console.print("[yellow]Falling back to default config.[/yellow]")
        return load_config()

    from questionary import Choice

    # Step 1: Preset?
    presets = list_available_configs().get("presets", [])
    use_preset = questionary.confirm("Start from a preset?").ask()
    preset_name = None
    if use_preset:
        preset_name = questionary.select("Choose preset:", choices=presets).ask()

    # Step 2: Model
    models = list_available_configs().get("models", [])
    model_name = questionary.select("Model:", choices=models + ["custom"]).ask()

    # Step 3: Attention
    try:
        from framework.attentions.core.registry import ATTENTION_REGISTRY
        att_keys = list(ATTENTION_REGISTRY.keys())
    except ImportError:
        att_keys = ["full"]
    attention = questionary.select("Attention type:", choices=att_keys).ask()

    # Step 4: Task
    tasks = list_available_configs().get("tasks", [])
    task = questionary.select("Task:", choices=tasks + ["custom"]).ask()

    # Step 5: Basic params
    epochs = int(questionary.text("Epochs:", default="3").ask())
    batch_size = int(questionary.text("Batch size:", default="16").ask())
    lr = float(questionary.text("Learning rate:", default="2e-5").ask())

    # Build
    paths = resolve_config_sources(
        task=task if task != "custom" else None,
        model=model_name if model_name != "custom" else None,
        attention=attention,
        preset=preset_name,
    )
    cfg = load_config(config_paths=paths, cli_overrides={
        "num_epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
    })
    return cfg


# ────────────────────────────────────────────────────────────────
# CLI group
# ────────────────────────────────────────────────────────────────

@click.group()
@click.version_option(version="2.0.0", prog_name="Efficient Transformer Toolkit")
def cli():
    """⚡ Efficient Transformer Toolkit – train, evaluate, compress, and more.

    Use pre‑built config profiles or create your own YAML files.
    """
    pass


# ────────────────────────────────────────────────────────────────
# train
# ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", "config_files", multiple=True, type=click.Path(exists=True),
              help="YAML config file(s). Can be specified multiple times.")
@click.option("--task", "-t", default=None, help="Task name (mrpc, mnli, sst2, cola, qqp, or custom file)")
@click.option("--model", "-m", "model_profile", default=None, help="Model profile (distilbert_base, bert_base, roberta_base)")
@click.option("--attention", "-a", default=None, help="Attention type (full, aurora, nova, prism_full, etc.)")
@click.option("--preset", "-p", default=None, help="Named preset (fast, accurate, extreme_compress)")
@click.option("--output", "-o", default=None, help="Output directory")
@click.option("--epochs", type=int, default=None, help="Number of epochs")
@click.option("--batch-size", type=int, default=None, help="Training batch size")
@click.option("--lr", type=float, default=None, help="Learning rate")
@click.option("--rank", type=int, default=None, help="Compression rank")
@click.option("--device", "-d", default=None, help="Device (cuda, cpu, auto)")
@click.option("--max-seq-length", type=int, default=None, help="Max sequence length")
@click.option("--use-unsloth", is_flag=True, default=None, help="Load via Unsloth")
@click.option("--interactive", "-i", is_flag=True, help="Interactive config builder")
@click.option("--print-config", "print_cfg", is_flag=True, help="Print resolved config and exit")
@click.option("--save-config", "save_cfg", type=click.Path(), default=None, help="Save resolved config to YAML")
def train(config_files, task, model_profile, attention, preset, output, epochs, batch_size,
          lr, rank, device, max_seq_length, use_unsloth, interactive, print_cfg, save_cfg):
    """Train a model with any attention mechanism on any dataset."""
    # ── Interactive mode ──
    if interactive:
        cfg = _interactive_config()
    else:
        overrides = {
            "output_dir": output,
            "num_epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "compression_rank": rank,
            "device": device,
            "max_seq_length": max_seq_length,
            "use_unsloth": use_unsloth,
        }
        # Remove None values
        overrides = {k: v for k, v in overrides.items() if v is not None}
        cfg = _build_config(
            task=task, model=model_profile, attention=attention,
            preset=preset, config_files=config_files, overrides=overrides,
        )

    # ── Print / Save config ──
    if print_cfg:
        _print_config(cfg)
    if save_cfg:
        cfg.to_yaml(Path(save_cfg))
        console.print(f"[green]Config saved to {save_cfg}[/green]")
    if print_cfg or save_cfg:
        return

    # ── Show config ──
    _print_config(cfg)

    # ── Load model ──
    console.print(f"\n[bold]Loading model: {cfg.model_name}[/bold]")
    from framework.models.model_loader import load_model
    model, tokenizer = load_model(
        model_name=cfg.model_name,
        num_labels=cfg.num_labels or 2,
        attention_type=cfg.attention_type,
        rank=cfg.compression_rank,
        device=cfg.device,
        use_unsloth=cfg.use_unsloth,
        cache_dir=cfg.cache_dir,
    )

    # ── Load dataset ──
    console.print(f"[bold]Loading dataset: {cfg.dataset_name}[/bold]")
    from framework.data.dataset_loader import ExperimentConfig, load_dataset
    data_cfg = ExperimentConfig(
        dataset_name=cfg.dataset_name,
        dataset_path=cfg.dataset_path,
        max_seq_length=cfg.max_seq_length,
        split_ratio=cfg.split_ratio,
        glue_base_path=cfg.glue_base_path,
        num_labels=cfg.num_labels or 2,
    )
    dataset_dict = load_dataset(data_cfg)

    # ── Auto‑detect num_labels for GLUE tasks ──
    from framework.data.glue_loader import get_num_labels_and_is_regression
    if cfg.dataset_name in {"cola","sst2","mrpc","qqp","stsb","mnli","qnli","rte","wnli"}:
        n_labels, is_reg = get_num_labels_and_is_regression(cfg.dataset_name)
        if cfg.num_labels is None:
            cfg.num_labels = n_labels

    # ── Tokenize ──
    console.print("[bold]Tokenizing...[/bold]")
    from framework.data.preprocessing import preprocess_function
    sent1_key, sent2_key = None, None
    pair_tasks = {"mrpc":("sentence1","sentence2"), "qqp":("question1","question2"),
                  "stsb":("sentence1","sentence2"), "mnli":("premise","hypothesis"),
                  "qnli":("question","sentence"), "rte":("sentence1","sentence2"),
                  "wnli":("sentence1","sentence2")}
    if cfg.dataset_name in pair_tasks:
        sent1_key, sent2_key = pair_tasks[cfg.dataset_name]
    elif cfg.dataset_name in {"cola","sst2"}:
        sent1_key = "sentence"

    preprocess_fn = preprocess_function(
        tokenizer=tokenizer, max_seq_length=cfg.max_seq_length,
        sent1_key=sent1_key, sent2_key=sent2_key,
        is_regression=(cfg.dataset_name=="stsb")
    )
    encoded = dataset_dict.map(preprocess_fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    # ── Train ──
    console.print(f"[bold green]Starting training ({cfg.num_epochs} epochs)...[/bold green]")
    from transformers import Trainer, TrainingArguments
    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        max_grad_norm=cfg.max_grad_norm,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_dir=f"{cfg.output_dir}/logs",
        logging_steps=cfg.logging_steps,
        save_total_limit=cfg.save_total_limit,
        seed=cfg.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=encoded["train"],
        eval_dataset=encoded.get("validation", encoded.get("test")),
    )
    trainer.train()

    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    console.print(f"[bold green]✅ Model saved to {cfg.output_dir}[/bold green]")


# ────────────────────────────────────────────────────────────────
# evaluate
# ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model-path", "-mp", required=True, help="Path to trained model")
@click.option("--task", "-t", default="mrpc", help="Task name")
@click.option("--max-seq-length", type=int, default=128)
@click.option("--batch-size", type=int, default=32)
@click.option("--device", default=None)
def evaluate(model_path, task, max_seq_length, batch_size, device):
    """Evaluate a trained model on a GLUE task."""
    if device is None:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device)

    from framework.data.dataset_loader import ExperimentConfig, load_dataset
    data_cfg = ExperimentConfig(dataset_name=task, max_seq_length=max_seq_length)
    dataset_dict = load_dataset(data_cfg)

    from framework.data.preprocessing import preprocess_function
    pair_tasks = {"mrpc":("sentence1","sentence2"), "qqp":("question1","question2"),
                  "stsb":("sentence1","sentence2"), "mnli":("premise","hypothesis"),
                  "qnli":("question","sentence"), "rte":("sentence1","sentence2"),
                  "wnli":("sentence1","sentence2")}
    sent1_key, sent2_key = None, None
    if task in pair_tasks:
        sent1_key, sent2_key = pair_tasks[task]
    elif task in {"cola","sst2"}:
        sent1_key = "sentence"

    fn = preprocess_function(tokenizer, max_seq_length=max_seq_length,
                             sent1_key=sent1_key, sent2_key=sent2_key,
                             is_regression=(task=="stsb"))
    encoded = dataset_dict.map(fn, batched=True)
    encoded.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    from transformers import Trainer, TrainingArguments
    args = TrainingArguments(output_dir="/tmp/eval", per_device_eval_batch_size=batch_size)
    trainer = Trainer(model=model, args=args,
                      eval_dataset=encoded.get("validation", encoded.get("test")))
    results = trainer.evaluate()
    console.print(Panel(str(results), title="Evaluation Results"))


# ────────────────────────────────────────────────────────────────
# compress
# ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--model", "-m", default="distilbert-base-uncased", help="Model name or path")
@click.option("--method", default="low_rank", help="Compression method (low_rank, pruning, quantization, hybrid)")
@click.option("--target-sparsity", type=float, default=0.5)
@click.option("--target-rank", type=int, default=64)
@click.option("--output", "-o", default="./compressed_model")
@click.option("--device", default=None)
def compress_cmd(model, method, target_sparsity, target_rank, output, device):
    """Compress a model using a registered compression method."""
    if device is None:
        import torch; device = "cuda" if torch.cuda.is_available() else "cpu"

    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    base_model = AutoModelForSequenceClassification.from_pretrained(model).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model)

    from framework.compression.core.registry import COMPRESSOR_REGISTRY
    compressor_cls = COMPRESSOR_REGISTRY.get(method)
    if compressor_cls is None:
        keys = list(COMPRESSOR_REGISTRY.keys())
        console.print(f"[red]Unknown method '{method}'. Available: {keys}[/red]")
        return

    compressor = compressor_cls(model=base_model, target_sparsity=target_sparsity,
                                target_rank=target_rank, method=method)
    compressed_model, stats = compressor.compress()

    table = Table(title="Compression Stats")
    table.add_column("Metric", style="cyan"); table.add_column("Value", style="green")
    for k, v in stats.items():
        if isinstance(v, (int, float, str)):
            table.add_row(k, str(v))
    console.print(table)

    compressed_model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    console.print(f"[green]✅ Compressed model saved to {output}[/green]")


# ────────────────────────────────────────────────────────────────
# generate‑synthetic
# ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "-o", default="./data/synthetic.csv")
@click.option("--samples", type=int, default=5000)
@click.option("--domain", default="movies", help="movies, tech, restaurants, books")
@click.option("--seed", type=int, default=42)
def generate_synthetic(output, samples, domain, seed):
    """Generate a synthetic classification dataset."""
    from framework.data.synthetic_generator import generate_synthetic_data
    generate_synthetic_data(output_path=output, num_samples=samples, domain=domain, seed=seed)


# ────────────────────────────────────────────────────────────────
# list‑configs
# ────────────────────────────────────────────────────────────────

@cli.command()
def list_configs():
    """List all available config profiles and attention types."""
    _show_available()


# ────────────────────────────────────────────────────────────────
# init (scaffold a config file)
# ────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--output", "-o", default="./my_experiment.yaml", help="Output config file path")
@click.option("--preset", "-p", default=None, help="Base preset")
def init_config(output, preset):
    """Create a new config file from defaults (or a preset)."""
    paths = [CONFIGS_DIR / "default.yaml"]
    if preset:
        p = CONFIGS_DIR / "presets" / f"{preset}.yaml"
        if p.exists():
            paths.append(p)
    cfg = load_config(config_paths=paths)
    cfg.to_yaml(Path(output))
    console.print(f"[green]Config scaffolded to {output}[/green]")
    _print_config(cfg)


if __name__ == "__main__":
    cli()