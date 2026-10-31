#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CLI Compress Command.

Implements the command-line interface for model compression using SVD,
low-rank factorization, quantization, and structured pruning.
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

import neuralops
from neuralops.core.registry import COMPRESSOR_REGISTRY

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    HAS_RICH = True
    console = Console()
except ImportError:
    HAS_RICH = False
    console = None

logger = logging.getLogger("NeuralOps.CLI.Compress")


def build_compress_parser(subparser: Optional[argparse.ArgumentParser] = None) -> argparse.ArgumentParser:
    """Construct argument parser for neuralops compress subcommand."""
    parser = subparser or argparse.ArgumentParser(
        prog="neuralops compress",
        description="Compress model weights using SVD, low-rank factorization, or pruning.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Target model identifier or local checkpoint path to compress",
    )
    parser.add_argument(
        "--compressor",
        "--method",
        dest="compressor",
        type=str,
        default="direct_svd",
        help="Target compression algorithm from registered compressors (default: 'direct_svd')",
    )
    parser.add_argument(
        "--rank",
        "--target-rank",
        dest="rank",
        type=int,
        default=32,
        help="Target rank for low-rank SVD / matrix factorizations (default: 32)",
    )
    parser.add_argument(
        "--target-ratio",
        "--ratio",
        dest="target_ratio",
        type=float,
        default=None,
        help="Rank fraction for direct_svd-family compressors (0 < ratio <= 1); not a guaranteed model-wide parameter ratio",
    )
    parser.add_argument(
        "--output-dir",
        "--save-dir",
        "--output",
        dest="output_dir",
        type=str,
        default="results/compressed",
        help="Destination directory to save compressed model checkpoint and manifests",
    )
    parser.add_argument(
        "--dataset",
        "--calibration-dataset",
        dest="dataset",
        type=str,
        default=None,
        help="Optional dataset or calibration data for data-dependent compression",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Target hardware device ('cuda', 'mps', 'cpu')",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output compression metrics as JSON",
    )
    return parser


def _load_model_for_compression(model_name_or_path: str, device: str) -> nn.Module:
    """Load a real pretrained model; never substitute random weights for a failed checkpoint."""
    errors: list[str] = []
    last_error: Exception | None = None
    try:
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        last_error = e
        errors.append(f"AutoModel: {e}")
        logger.debug("AutoModel loading failed: %s", e)

    try:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        model.to(device)
        return model
    except Exception as e:
        last_error = e
        errors.append(f"AutoModelForCausalLM: {e}")
        logger.debug("AutoModelForCausalLM loading failed: %s", e)
    detail = "; ".join(errors)
    raise RuntimeError(
        f"Could not load checkpoint {model_name_or_path!r} as a supported Hugging Face model. "
        f"No placeholder model was created. Loader errors: {detail}"
    ) from last_error


def cmd_compress(args: Any) -> int:
    """Execute model compression via registered compressor engines."""
    if isinstance(args, dict):
        args_ns = argparse.Namespace(**args)
    else:
        args_ns = args

    model_id = getattr(args_ns, "model", None)
    if not model_id:
        print("[-] Error: --model is required for compression.", file=sys.stderr)
        return 1

    compressor_name = getattr(args_ns, "compressor", None) or getattr(args_ns, "method", "direct_svd")
    rank = getattr(args_ns, "rank", 32)
    target_ratio = getattr(args_ns, "target_ratio", None)
    output_dir = getattr(args_ns, "output_dir", None) or getattr(args_ns, "save_dir", "results/compressed")
    dataset = getattr(args_ns, "dataset", None) or getattr(args_ns, "calibration_dataset", None)
    device_arg = getattr(args_ns, "device", None)
    output_json = getattr(args_ns, "json", False)

    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 1:
        raise ValueError("--rank must be a positive integer")
    if target_ratio is not None and (
        isinstance(target_ratio, bool)
        or not isinstance(target_ratio, (int, float))
        or not math.isfinite(target_ratio)
        or not 0.0 < target_ratio <= 1.0
    ):
        raise ValueError("--target-ratio must be in the interval (0, 1]")
    COMPRESSOR_REGISTRY.resolve(compressor_name)
    if target_ratio is not None and compressor_name.lower() not in {
        "direct_svd", "svd", "hnc_svd", "hnc_svd_compressor"
    }:
        raise ValueError(
            f"--target-ratio is only supported by direct_svd-family compressors; "
            f"{compressor_name!r} does not declare rank-ratio support."
        )

    if device_arg:
        device = device_arg
    else:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    if not output_json:
        if HAS_RICH and console is not None:
            console.print(Panel(
                f"[bold magenta]NeuralOps Model Compression Engine[/bold magenta]\n"
                f"[bold white]Target Model:[/]  [yellow]{model_id}[/]\n"
                f"[bold white]Compressor:[/]    [green]{compressor_name}[/]\n"
                f"[bold white]Target Rank:[/]   {rank}  [bold white]| Rank Fraction:[/] {target_ratio or 'auto'}\n"
                f"[bold white]Device:[/]        [cyan]{device.upper()}[/]  [bold white]| Save Dir:[/] [blue]{output_dir}[/]",
                title="[bold magenta]NeuralOps Compression Suite[/bold magenta]",
                border_style="magenta",
            ))
        else:
            print("=================================================================")
            print(f"NeuralOps Model Compression: {model_id} via {compressor_name}")
            print(f"Rank: {rank} | Rank Fraction: {target_ratio} | Device: {device}")
            print("=================================================================")

    # 1. Load base model
    if not output_json:
        print(f"[*] Loading model '{model_id}' on {device}...")
    model = _load_model_for_compression(model_id, device=device)

    orig_params = sum(p.numel() for p in model.parameters())
    orig_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    # 2. Invoke compression
    if not output_json:
        print(f"[*] Executing compression algorithm '{compressor_name}' (rank={rank})...")
    compress_kwargs: Dict[str, Any] = {}
    if target_ratio is not None:
        compress_kwargs["rank_ratio"] = target_ratio
    if dataset is not None:
        compress_kwargs["dataset"] = dataset

    start_time = time.perf_counter()
    try:
        compressed_model = neuralops.compress(
            model=model,
            method=compressor_name,
            rank=rank,
            **compress_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Compression with {compressor_name!r} failed; no alternate compressor was substituted "
            "and no output artifact was written."
        ) from exc
    if not isinstance(compressed_model, nn.Module):
        raise TypeError(
            f"Compressor {compressor_name!r} returned {type(compressed_model).__name__}, "
            "expected torch.nn.Module. No output artifact was written."
        )

    elapsed = time.perf_counter() - start_time

    comp_params = sum(p.numel() for p in compressed_model.parameters())
    comp_bytes = sum(p.numel() * p.element_size() for p in compressed_model.parameters())

    param_ratio = (orig_params / max(1, comp_params))
    param_savings_pct = max(0.0, (1.0 - (comp_params / max(1, orig_params))) * 100.0)

    # Singular value truncation error / spectral norm residual
    stats = getattr(compressed_model, "_compression_stats", {})
    if not isinstance(stats, dict):
        stats = {}
    trunc_error = stats.get("truncation_error", stats.get("avg_reconstruction_error"))

    # 3. Save compressed model
    os.makedirs(output_dir, exist_ok=True)
    serialization_format = "state_dict"
    if callable(getattr(compressed_model, "save_pretrained", None)):
        compressed_model.save_pretrained(output_dir)
        serialization_format = "huggingface_pretrained+state_dict"
    model_save_path = os.path.join(output_dir, "compressed_model.pt")
    torch.save(compressed_model.state_dict(), model_save_path)

    manifest: Dict[str, Any] = {
        "model": model_id,
        "compressor": compressor_name,
        "rank": rank,
        "rank_ratio": target_ratio,
        "original_parameters": orig_params,
        "compressed_parameters": comp_params,
        "compression_ratio": round(param_ratio, 2),
        "parameter_savings_percent": round(param_savings_pct, 2),
        "reconstruction_error": round(float(trunc_error), 6) if trunc_error is not None else None,
        "compressed_layer_count": stats.get("compressed_layer_count"),
        "fallback_layer_count": stats.get("fallback_layer_count"),
        "skipped_layer_count": stats.get("skipped_layer_count"),
        "original_size_mb": round(orig_bytes / (1024 * 1024), 2),
        "compressed_size_mb": round(comp_bytes / (1024 * 1024), 2),
        "compression_time_seconds": round(elapsed, 3),
        "serialization_format": serialization_format,
        "output_dir": output_dir,
    }

    manifest_path = os.path.join(output_dir, "compression_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    if output_json:
        print(json.dumps(manifest, indent=2))
    else:
        if HAS_RICH and console is not None:
            table = Table(title="Compression Metrics & Artifacts", header_style="bold green")
            table.add_column("Property", style="cyan", width=28)
            table.add_column("Value", style="bold yellow", min_width=25)
            table.add_column("Notes", style="dim", width=20)

            table.add_row("Base Parameters", f"{orig_params:,}", "pre-compression")
            table.add_row("Compressed Parameters", f"{comp_params:,}", "post-compression")
            table.add_row("Effective Ratio", f"{param_ratio:.2f}x", "parameter scaling")
            table.add_row("Parameter Savings", f"{param_savings_pct:.2f}%", "memory footprint")
            table.add_row(
                "Reconstruction Error",
                f"{trunc_error:.6f}" if trunc_error is not None else "not reported",
                "compressor-reported",
            )
            table.add_row("Compression Latency", f"{elapsed:.3f} s", "execution duration")
            table.add_row("Artifact Destination", str(output_dir), "saved state_dict")

            console.print(table)
            console.print(f"[bold green]✓ Compressed model and manifest saved to '{output_dir}'.[/]")
        else:
            print("-----------------------------------------------------------------")
            print(f"  Base Parameters:         {orig_params:,}")
            print(f"  Compressed Parameters:   {comp_params:,}")
            print(f"  Compression Ratio:       {param_ratio:.2f}x")
            print(f"  Parameter Savings:       {param_savings_pct:.2f}%")
            error_text = f"{trunc_error:.6f}" if trunc_error is not None else "not reported by compressor"
            print(f"  Reconstruction Error:    {error_text}")
            print(f"  Duration:                {elapsed:.3f} s")
            print(f"  Saved to:                {output_dir}")
            print("-----------------------------------------------------------------")

    return 0


if __name__ == "__main__":
    parser = build_compress_parser()
    cli_args = parser.parse_args()
    sys.exit(cmd_compress(cli_args))
