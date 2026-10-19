#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Train Command.

Implements the command-line interface for model pretraining, fine-tuning,
multi-corpus variable-length packing, and scheduler execution.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralops.data.multidataset import MultiDatasetLoader
from neuralops.data.packing import PackedCollator, PackedDataset
from neuralops.trainer.optimizers import (
    create_adamw,
    create_lion,
    create_muon,
    create_sgd,
)
from neuralops.trainer.schedulers import get_lr_scheduler

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger("NeuralOps.CLI.Train")


def build_train_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops train subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops train",
        description="Fine-tune or pretrain models with variable-length packing and custom schedules.",
    )
    # Model specification
    parser.add_argument(
        "--model",
        type=str,
        default="gpt2",
        help="Model identifier or local checkpoint path (e.g. 'gpt2', 'distilbert-base-uncased', 'llama')",
    )
    # Dataset & multi-corpus specification
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Target dataset or comma-separated multi-datasets (e.g. 'HuggingFaceFW/fineweb-edu,opencsg/Fineweb-Edu-Chinese-V2')",
    )
    parser.add_argument(
        "--dataset-name",
        "--subset",
        dest="dataset_name",
        type=str,
        default=None,
        help="Dataset subset/configuration name, or comma-separated subset names",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to train on (default: 'train')",
    )
    # Varlen packing flags
    parser.add_argument(
        "--varlen",
        action="store_true",
        help="Enable variable-length document packing without padding token waste",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=4096,
        help="Packed sequence length (e.g. 4096, 8192, 65536)",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=4096,
        help="Maximum individual document or chunk length (default: 4096)",
    )
    # Hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size per training device (default: 4)",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        "--grad-accum",
        dest="gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before backward/step (default: 1)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Peak learning rate (default: 1e-4)",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default="wsd",
        choices=["wsd", "cosine", "linear"],
        help="Learning rate schedule policy: 'wsd', 'cosine', or 'linear' (default: 'wsd')",
    )
    parser.add_argument(
        "--decay-ratio",
        type=float,
        default=0.2,
        help="Ratio of total steps allocated to terminal decay phase in WSD schedule (default: 0.2)",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="Linear warmup steps for optimizer (default: 100)",
    )
    parser.add_argument(
        "--lr-min",
        type=float,
        default=1e-6,
        help="Minimum terminal learning rate (default: 1e-6)",
    )
    # Optimizers
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adamw",
        choices=["adamw", "lion", "muon", "sgd"],
        help="Optimizer formulation: 'adamw', 'lion', 'muon', or 'sgd' (default: 'adamw')",
    )
    # Execution controls
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Number of complete training passes over dataset (default: 1)",
    )
    parser.add_argument(
        "--steps",
        "--max-steps",
        dest="steps",
        type=int,
        default=None,
        help="Maximum optimization steps before early termination",
    )
    parser.add_argument(
        "--save-dir",
        "--output-dir",
        dest="save_dir",
        type=str,
        default="results/trained",
        help="Destination directory to save model weights and manifests",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=500,
        help="Interval of steps at which intermediate checkpoints are saved",
    )
    parser.add_argument(
        "--eval-steps",
        type=int,
        default=250,
        help="Evaluation step frequency during training runs",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile model using torch.compile() for accelerated kernel execution",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML or JSON configuration file overriding CLI defaults",
    )
    return parser


def _load_model(model_name_or_path: str, device: str) -> nn.Module:
    """Load model architecture or construct lightweight fallback model."""
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        logger.warning("AutoModelForCausalLM loading failed for '%s': %s", model_name_or_path, e)

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        logger.warning("AutoModel loading failed for '%s': %s. Using robust neural architecture.", model_name_or_path, e)

    # Lightweight fallback model for test/simulation
    class _FallbackLanguageModel(nn.Module):
        def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512, num_layers: int = 4) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=8, dim_feedforward=hidden_dim * 4, batch_first=True)
                for _ in range(num_layers)
            ])
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
            x = self.embed(input_ids)
            for layer in self.layers:
                x = layer(x)
            logits = self.head(x)
            loss = None
            if labels is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )
            class Output:
                def __init__(self, loss_val: Optional[torch.Tensor], logits_val: torch.Tensor) -> None:
                    self.loss = loss_val
                    self.logits = logits_val
            return Output(loss, logits)

    model = _FallbackLanguageModel()
    model.to(device)
    return model


def _create_optimizer(optimizer_name: str, model: nn.Module, lr: float) -> torch.optim.Optimizer:
    """Instantiate requested optimizer from registered algorithms."""
    name = optimizer_name.lower()
    if name == "lion":
        return create_lion(model.parameters(), lr=lr, weight_decay=0.01)
    elif name == "muon":
        return create_muon(model.parameters(), lr=lr)
    elif name == "sgd":
        return create_sgd(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
    else:
        return create_adamw(model.parameters(), lr=lr, weight_decay=0.01)


def cmd_train(args: Any) -> int:
    """Execute training pipeline with variable-length packing and scheduling."""
    # Handle dictionary or namespace
    if isinstance(args, dict):
        args_ns = argparse.Namespace(**args)
    else:
        args_ns = args

    # Check for external configuration file
    config_path = getattr(args_ns, "config", None)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                if config_path.endswith(".json"):
                    cfg = json.load(f)
                else:
                    import yaml

                    cfg = yaml.safe_load(f)
                for k, v in cfg.items():
                    setattr(args_ns, k.replace("-", "_"), v)
        except Exception as e:
            logger.warning("Could not parse config file '%s': %s", config_path, e)

    model_id = getattr(args_ns, "model", "gpt2")
    dataset_spec = getattr(args_ns, "dataset", "wikitext")
    dataset_name = getattr(args_ns, "dataset_name", None)
    split = getattr(args_ns, "split", "train")
    varlen = getattr(args_ns, "varlen", False)
    seq_len = getattr(args_ns, "seq_len", 4096)
    context_len = getattr(args_ns, "context_len", 4096)
    batch_size = getattr(args_ns, "batch_size", 4)
    grad_accum = getattr(args_ns, "gradient_accumulation_steps", 1)
    lr = getattr(args_ns, "lr", 1e-4)
    lr_scheduler_type = getattr(args_ns, "lr_scheduler", "wsd")
    decay_ratio = getattr(args_ns, "decay_ratio", 0.2)
    warmup_steps = getattr(args_ns, "warmup_steps", 100)
    lr_min = getattr(args_ns, "lr_min", 1e-6)
    optimizer_name = getattr(args_ns, "optimizer", "adamw")
    epochs = getattr(args_ns, "epochs", 1)
    max_steps = getattr(args_ns, "steps", None) or getattr(args_ns, "max_steps", None)
    save_dir = getattr(args_ns, "save_dir", "results/trained") or getattr(args_ns, "output_dir", "results/trained")
    checkpoint_interval = getattr(args_ns, "checkpoint_interval", 500)
    eval_steps = getattr(args_ns, "eval_steps", 250)
    compile_model = getattr(args_ns, "compile", False)

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    # Display execution banner
    if HAS_RICH and console is not None:
        table = Table(title="NeuralOps Training Run Configuration", header_style="bold magenta")
        table.add_column("Parameter", style="cyan", width=26)
        table.add_column("Setting", style="bold green", min_width=35)
        table.add_row("Model Architecture", str(model_id))
        table.add_row("Dataset Spec", str(dataset_spec))
        table.add_row("Subset / Domain", str(dataset_name or "default"))
        table.add_row("Varlen Packing", "[bold green]Active[/]" if varlen else "[dim]Disabled (Fixed)[/]")
        table.add_row("Packed Seq Length", f"{seq_len} tokens")
        table.add_row("Document Context Length", f"{context_len} tokens")
        table.add_row("Optimizer Formulation", optimizer_name.upper())
        table.add_row("Learning Rate Scheduler", f"{lr_scheduler_type.upper()} (Decay: {decay_ratio:.2f})")
        table.add_row("Peak Learning Rate", f"{lr:.2e}")
        table.add_row("Batch Size (Per Device)", str(batch_size))
        table.add_row("Gradient Accumulation", f"{grad_accum} steps")
        table.add_row("Target Device", device.upper())
        table.add_row("torch.compile()", "Enabled" if compile_model else "Disabled")
        console.print(table)
    else:
        print("=================================================================")
        print("NeuralOps Training Engine")
        print(f"Model: {model_id} | Dataset: {dataset_spec} | Varlen: {varlen}")
        print(f"Optimizer: {optimizer_name} | Scheduler: {lr_scheduler_type} | LR: {lr}")
        print(f"Batch Size: {batch_size} | Device: {device} | SeqLen: {seq_len}")
        print("=================================================================")

    # 1. Load model
    print(f"[*] Loading model '{model_id}' on {device}...")
    model = _load_model(model_id, device=device)
    if compile_model and hasattr(torch, "compile"):
        print("[*] JIT compiling model with torch.compile()...")
        try:
            model = torch.compile(model)  # type: ignore[assignment]
        except Exception as e:
            logger.warning("torch.compile failed: %s. Continuing with eager execution.", e)

    # 2. Build multi-dataset loader
    print(f"[*] Preparing dataset streams for '{dataset_spec}'...")
    multi_loader = MultiDatasetLoader(
        dataset_names=dataset_spec,
        subsets=dataset_name,
        split=split,
        streaming=True,
    )

    # 3. Construct packed dataset or standard dataloader
    if varlen or "," in str(dataset_spec):
        print(f"[*] Initializing PackedDataset (seq_len={seq_len}, context_len={context_len})...")
        dataset_wrapper = PackedDataset(
            dataset=multi_loader,
            seq_len=seq_len,
            context_len=context_len,
        )
        collator = PackedCollator()
        train_dataloader = DataLoader(
            dataset_wrapper,
            batch_size=batch_size,
            collate_fn=collator,
        )
    else:
        train_dataloader = DataLoader(
            multi_loader,
            batch_size=batch_size,
        )

    # 4. Construct optimizer and scheduler
    optimizer = _create_optimizer(optimizer_name, model, lr=lr)
    total_training_steps = max_steps or 1000
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        scheduler_type=lr_scheduler_type,
        total_steps=total_training_steps,
        warmup_steps=warmup_steps,
        decay_ratio=decay_ratio,
        lr_min=lr_min,
    )

    os.makedirs(save_dir, exist_ok=True)
    step = 0
    running_loss = 0.0
    accum_loss = 0.0
    total_tokens_processed = 0
    start_time = time.time()

    print("[*] Initiating training loop...")
    model.train()
    optimizer.zero_grad()

    try:
        for epoch in range(epochs):
            for batch in train_dataloader:
                step += 1
                if isinstance(batch, dict):
                    input_ids = batch.get("input_ids")
                    labels = batch.get("labels", input_ids)
                else:
                    input_ids = batch
                    labels = batch

                if input_ids is None:
                    continue

                if isinstance(input_ids, torch.Tensor):
                    input_ids = input_ids.to(device)
                else:
                    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)

                if isinstance(labels, torch.Tensor):
                    labels = labels.to(device)
                else:
                    labels = torch.tensor(labels, dtype=torch.long, device=device)

                total_tokens_processed += input_ids.numel()

                outputs = model(input_ids=input_ids, labels=labels)
                loss = getattr(outputs, "loss", None)
                if loss is None:
                    logits = getattr(outputs, "logits", outputs)
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = labels[..., 1:].contiguous()
                    loss = nn.functional.cross_entropy(
                        shift_logits.view(-1, shift_logits.size(-1)),
                        shift_labels.view(-1),
                        ignore_index=-100,
                    )

                loss = loss / grad_accum
                loss.backward()
                accum_loss += loss.item() * grad_accum

                if step % grad_accum == 0:
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad()
                    running_loss = accum_loss / grad_accum
                    accum_loss = 0.0

                if step % 25 == 0 or step == 1:
                    elapsed = max(1e-4, time.time() - start_time)
                    tok_per_sec = total_tokens_processed / elapsed
                    current_lr = optimizer.param_groups[0]["lr"]
                    print(
                        f"[Step {step:05d}/{total_training_steps}] "
                        f"Loss: {running_loss:.4f} | LR: {current_lr:.3e} | "
                        f"Throughput: {tok_per_sec:,.0f} tok/s"
                    )

                if step % checkpoint_interval == 0:
                    ckpt_path = os.path.join(save_dir, f"checkpoint-step-{step}")
                    os.makedirs(ckpt_path, exist_ok=True)
                    if hasattr(model, "save_pretrained"):
                        try:
                            model.save_pretrained(ckpt_path)
                        except Exception:
                            torch.save(model.state_dict(), os.path.join(ckpt_path, "pytorch_model.bin"))
                    else:
                        torch.save(model.state_dict(), os.path.join(ckpt_path, "pytorch_model.bin"))
                    print(f"[✓] Checkpoint saved: {ckpt_path}")

                if max_steps is not None and step >= max_steps:
                    break

            if max_steps is not None and step >= max_steps:
                break
    except KeyboardInterrupt:
        print("\n[!] Training interrupted by user. Saving final state...")

    # Final checkpoint & manifest
    elapsed_total = max(1e-4, time.time() - start_time)
    avg_tokens_sec = total_tokens_processed / elapsed_total
    final_ckpt = os.path.join(save_dir, "final_model")
    os.makedirs(final_ckpt, exist_ok=True)
    if hasattr(model, "save_pretrained"):
        try:
            model.save_pretrained(final_ckpt)
        except Exception:
            torch.save(model.state_dict(), os.path.join(final_ckpt, "pytorch_model.bin"))
    else:
        torch.save(model.state_dict(), os.path.join(final_ckpt, "pytorch_model.bin"))

    manifest = {
        "model": model_id,
        "dataset": dataset_spec,
        "optimizer": optimizer_name,
        "lr_scheduler": lr_scheduler_type,
        "total_steps": step,
        "total_tokens": total_tokens_processed,
        "tokens_per_second": avg_tokens_sec,
        "elapsed_seconds": elapsed_total,
        "final_loss": running_loss,
        "save_dir": save_dir,
    }
    with open(os.path.join(save_dir, "training_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[✓] Training complete. Processed {total_tokens_processed:,} tokens in {elapsed_total:.2f}s ({avg_tokens_sec:,.0f} tok/s).")
    print(f"[✓] Checkpoint and manifest written to '{save_dir}'.")
    return 0


if __name__ == "__main__":
    parser = build_train_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_train(cli_args))
