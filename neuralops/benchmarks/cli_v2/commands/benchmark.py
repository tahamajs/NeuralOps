"""Benchmark subcommand for CLI v2."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

from neuralops.benchmarks.cli_v2.utils.rendering import print_cli_error, render_badge


def setup_benchmark_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
) -> argparse.ArgumentParser:
    """Setup parser for benchmark subcommand.

    Args:
        subparsers: Subparsers action from the parent parser.

    Returns:
        Configured subparser for the benchmark command.
    """
    parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML/JSON benchmark configuration")
    parser.add_argument("--modules", type=str, default="all", help="Modules to benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda, etc.)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Precision")
    parser.add_argument("--dry-run", action="store_true", help="Dry run estimation without heavy execution")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging output")
    return parser


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark run with device environment configuration.

    Args:
        args: Parsed arguments namespace.

    Returns:
        Exit code (0 for success, non-zero for failure).
    """
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""

    config_path = getattr(args, "config", None)
    if config_path:
        cfg_file = Path(config_path)
        if not cfg_file.is_file():
            print_cli_error(
                f"Benchmark configuration file not found: {config_path}",
                hint="Check the path and ensure the YAML or JSON config file exists.",
            )
            return 1

        try:
            from neuralops.benchmarks.config import BenchmarkConfig

            config = BenchmarkConfig.from_file(str(cfg_file))
            if hasattr(config, "device") and config.device:
                args.device = config.device
                if args.device == "cpu":
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
            if hasattr(config, "batch_size"):
                args.batch_size = config.batch_size
            if hasattr(config, "warmup"):
                args.warmup = config.warmup
            if hasattr(config, "iterations"):
                args.iterations = config.iterations
        except Exception as exc:  # allowed-swallow: config loading errors reported via CLI error
            print_cli_error(f"Failed to load benchmark configuration: {exc}")
            return 1

    if getattr(args, "dry_run", False):
        print(f"Benchmark dry-run: modules={getattr(args, 'modules', 'all')}, device={getattr(args, 'device', 'auto')}")
        return 0

    return 0
