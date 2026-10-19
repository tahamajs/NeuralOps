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
    seq_len = getattr(args_ns, "seq_len", 512)
    max_samples = getattr(args_ns, "max_samples", 100)
    device_arg = getattr(args_ns, "device", None)
    metrics_arg = getattr(args_ns, "metrics", "perplexity,throughput,latency,peak_vram")
    output_json = getattr(args_ns, "output_json", False) or getattr(args_ns, "json", False)
    output_file = getattr(args_ns, "output_file", None)

    if device_arg:
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if not output_json:
        if HAS_RICH and console is not None:
            console.print(Panel(
                f"[bold cyan]NeuralOps Unified Evaluation Runner[/bold cyan]\n"
                f"[bold white]Model:[/]     [yellow]{model_id}[/]\n"
                f"[bold white]Dataset:[/]   [green]{dataset}[/] ({dataset_name or 'default'}, {split})\n"
                f"[bold white]Batch Size:[/] {batch_size}  [bold white]| Samples:[/] {max_samples}  [bold white]| Device:[/] [magenta]{str(device).upper()}[/]",
                title="[bold cyan]NeuralOps Evaluation & Benchmark Suite[/bold cyan]",
                border_style="cyan",
            ))
        else:
            print("=================================================================")
            print(f"NeuralOps Model Evaluation: {model_id} on {dataset} ({device})")
            print("=================================================================")

    from neuralops.benchmarks.eval_bridge import BenchmarkEvaluator, BenchmarkEvalResult

    evaluator = BenchmarkEvaluator(
        model=model_id,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
    )

    metrics_list = [m.strip().lower() for m in metrics_arg.split(",") if m.strip()]
    eval_result: BenchmarkEvalResult = evaluator.run_evaluation(
        dataset=dataset,
        subset=dataset_name,
        split=split,
        max_samples=max_samples,
        metrics=metrics_list,
    )

    results_data: Dict[str, Any] = eval_result.to_dict()
    results_data.update({
        "model": model_id,
        "dataset": dataset,
        "split": split,
        "samples_evaluated": eval_result.samples_evaluated,
        "total_tokens": eval_result.total_tokens,
        "elapsed_seconds": eval_result.duration_seconds,
        "loss": eval_result.loss,
        "perplexity": eval_result.perplexity,
        "throughput_tokens_per_sec": eval_result.throughput_tokens_per_sec,
        "latency_ms_per_token": eval_result.latency_ms_per_token,
        "peak_vram_mb": eval_result.peak_vram_mb,
        "accuracy_pct": eval_result.accuracy_pct,
        "device": str(evaluator.device),
    })

    avg_loss = eval_result.loss or 0.0
    perplexity = eval_result.perplexity or 0.0
    throughput_tokens_sec = eval_result.throughput_tokens_per_sec
    latency_ms_per_token = eval_result.latency_ms_per_token
    peak_vram_mb = eval_result.peak_vram_mb
    accuracy_pct = eval_result.accuracy_pct or 0.0
    total_tokens = eval_result.total_tokens
    elapsed_time = eval_result.duration_seconds

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
