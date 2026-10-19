#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Evaluation Command.

Implements the command-line interface for model evaluation across benchmark datasets,
measuring perplexity, token throughput, latency, and memory utilization.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from neuralops.data.multidataset import MultiDatasetLoader

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger("NeuralOps.CLI.Eval")


def build_eval_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops eval subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops eval",
        description="Benchmark model perplexity, token throughput, latency, and peak memory.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model identifier or local checkpoint path to evaluate",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="wikitext",
        help="Target evaluation dataset (e.g. 'wikitext', 'sst2', 'c4', 'glue')",
    )
    parser.add_argument(
        "--dataset-name",
        "--subset",
        dest="dataset_name",
        type=str,
        default=None,
        help="Dataset subset/configuration name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="validation",
        help="Dataset split to evaluate (default: 'validation')",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Evaluation batch size per forward pass (default: 16)",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=512,
        help="Context sequence length (default: 512)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=100,
        help="Maximum number of sequences / samples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Execution hardware device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="perplexity,throughput,latency,peak_vram",
        help="Comma-separated metrics to evaluate ('perplexity', 'accuracy', 'latency', 'throughput')",
    )
    parser.add_argument(
        "--json",
        "--output-json",
        dest="output_json",
        action="store_true",
        help="Format evaluation output as JSON string",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Destination file path to save evaluation results JSON",
    )
    return parser


def _load_eval_model(model_name_or_path: str, device: str) -> nn.Module:
    """Load model architecture or construct lightweight fallback model."""
    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        logger.debug("AutoModelForCausalLM loading failed for '%s': %s", model_name_or_path, e)

    try:
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        logger.debug("AutoModelForSequenceClassification loading failed for '%s': %s", model_name_or_path, e)

    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        logger.debug("AutoModel loading failed: %s", e)

    # Lightweight fallback model for test/simulation
    class _FallbackEvalModel(nn.Module):
        def __init__(self, vocab_size: int = 32000, hidden_dim: int = 512) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_dim)
            self.linear = nn.Linear(hidden_dim, hidden_dim)
            self.head = nn.Linear(hidden_dim, vocab_size, bias=False)

        def forward(self, input_ids: torch.Tensor, labels: Optional[torch.Tensor] = None, **kwargs: Any) -> Any:
            x = self.embed(input_ids)
            x = self.linear(x)
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

    model = _FallbackEvalModel()
    model.to(device)
    return model


def cmd_eval(args: Any) -> int:
    """Execute model evaluation across perplexity, throughput, latency, and memory."""
    if isinstance(args, dict):
        args_ns = argparse.Namespace(**args)
    else:
        args_ns = args

    model_id = getattr(args_ns, "model", "gpt2")
    dataset = getattr(args_ns, "dataset", "wikitext")
    dataset_name = getattr(args_ns, "dataset_name", None)
    split = getattr(args_ns, "split", "validation")
    batch_size = getattr(args_ns, "batch_size", 16)
    max_samples = getattr(args_ns, "max_samples", 100)
    device_arg = getattr(args_ns, "device", None)
    output_json = getattr(args_ns, "output_json", False) or getattr(args_ns, "json", False)
    output_file = getattr(args_ns, "output_file", None)

    if device_arg:
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if not output_json:
        if HAS_RICH and console is not None:
            console.print(Panel(
                f"[bold cyan]NeuralOps Evaluation Runner[/bold cyan]\n"
                f"[bold white]Model:[/]     [yellow]{model_id}[/]\n"
                f"[bold white]Dataset:[/]   [green]{dataset}[/] ({dataset_name or 'default'}, {split})\n"
                f"[bold white]Batch Size:[/] {batch_size}  [bold white]| Samples:[/] {max_samples}  [bold white]| Device:[/] [magenta]{device.upper()}[/]",
                title="[bold cyan]NeuralOps Evaluation Suite[/bold cyan]",
                border_style="cyan",
            ))
        else:
            print("=================================================================")
            print(f"NeuralOps Model Evaluation: {model_id} on {dataset} ({device})")
            print("=================================================================")

    # 1. Load model
    model = _load_eval_model(model_id, device=device)
    model.eval()

    # 2. Load dataset stream
    loader = MultiDatasetLoader(
        dataset_names=dataset,
        subsets=dataset_name,
        split=split,
        streaming=True,
    )
    dataloader = DataLoader(loader, batch_size=batch_size)

    # 3. Track metrics
    total_loss = 0.0
    total_batches = 0
    total_tokens = 0
    samples_evaluated = 0
    correct_tokens = 0
    evaluated_tokens_for_acc = 0

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    with torch.no_grad():
        for batch in dataloader:
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

            num_tokens = input_ids.numel()
            total_tokens += num_tokens
            samples_evaluated += input_ids.shape[0]

            outputs = model(input_ids=input_ids, labels=labels)
            loss = getattr(outputs, "loss", None)
            logits = getattr(outputs, "logits", outputs if isinstance(outputs, torch.Tensor) else None)

            if loss is None and logits is not None:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                    ignore_index=-100,
                )

            if loss is not None:
                total_loss += loss.item()
                total_batches += 1

            if logits is not None and labels is not None:
                preds = torch.argmax(logits, dim=-1)
                valid_mask = labels != -100
                correct = (preds == labels) & valid_mask
                correct_tokens += int(correct.sum().item())
                evaluated_tokens_for_acc += int(valid_mask.sum().item())

            if max_samples is not None and samples_evaluated >= max_samples:
                break

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    else:
        peak_vram_mb = 0.0

    elapsed_time = max(1e-5, time.perf_counter() - start_time)

    avg_loss = total_loss / max(1, total_batches) if total_batches > 0 else 0.0
    perplexity = math.exp(min(avg_loss, 80.0))
    throughput_tokens_sec = total_tokens / elapsed_time
    latency_ms_per_token = (elapsed_time * 1000.0) / max(1, total_tokens)
    accuracy_pct = (correct_tokens / max(1, evaluated_tokens_for_acc)) * 100.0 if evaluated_tokens_for_acc > 0 else 0.0

    results_data: Dict[str, Any] = {
        "model": model_id,
        "dataset": dataset,
        "split": split,
        "samples_evaluated": samples_evaluated,
        "total_tokens": total_tokens,
        "elapsed_seconds": round(elapsed_time, 4),
        "loss": round(avg_loss, 4),
        "perplexity": round(perplexity, 4),
        "throughput_tokens_per_sec": round(throughput_tokens_sec, 2),
        "latency_ms_per_token": round(latency_ms_per_token, 4),
        "peak_vram_mb": round(peak_vram_mb, 2),
        "accuracy_pct": round(accuracy_pct, 2),
        "device": device,
    }

    if output_json:
        print(json.dumps(results_data, indent=2))
    else:
        if HAS_RICH and console is not None:
            table = Table(title=f"Evaluation Results: {model_id}", header_style="bold green")
            table.add_column("Metric", style="cyan", width=28)
            table.add_column("Score / Reading", style="bold yellow", min_width=24)
            table.add_column("Unit", style="dim", width=16)

            table.add_row("Evaluation Loss", f"{avg_loss:.4f}", "cross-entropy")
            table.add_row("Perplexity (PPL)", f"{perplexity:.4f}", "exp(loss)")
            table.add_row("Token Throughput", f"{throughput_tokens_sec:,.2f}", "tokens / sec")
            table.add_row("Token Latency", f"{latency_ms_per_token:.4f}", "ms / token")
            table.add_row("Peak VRAM Allocated", f"{peak_vram_mb:.2f}", "MB")
            if evaluated_tokens_for_acc > 0:
                table.add_row("Next-Token Accuracy", f"{accuracy_pct:.2f}%", "percentage")
            table.add_row("Total Evaluated Tokens", f"{total_tokens:,}", "tokens")
            table.add_row("Evaluation Duration", f"{elapsed_time:.3f}", "seconds")

            console.print(table)
        else:
            print("-----------------------------------------------------------------")
            print(f"  Perplexity (PPL):        {perplexity:.4f}")
            print(f"  Throughput:              {throughput_tokens_sec:,.2f} tokens/sec")
            print(f"  Latency:                 {latency_ms_per_token:.4f} ms/token")
            print(f"  Peak VRAM:               {peak_vram_mb:.2f} MB")
            print(f"  Loss:                    {avg_loss:.4f}")
            if evaluated_tokens_for_acc > 0:
                print(f"  Accuracy:                {accuracy_pct:.2f}%")
            print(f"  Total Tokens:            {total_tokens:,} in {elapsed_time:.2f}s")
            print("-----------------------------------------------------------------")

    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as fp:
            json.dump(results_data, fp, indent=2)
        if not output_json:
            print(f"[✓] Results saved to '{output_file}'.")

    return 0


if __name__ == "__main__":
    parser = build_eval_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_eval(cli_args))
