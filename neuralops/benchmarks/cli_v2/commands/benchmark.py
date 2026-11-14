"""Benchmark subcommand for CLI v2."""

import argparse
import os
import sys
from typing import List, Optional


def setup_benchmark_parser(subparsers: argparse._SubParsersAction) -> argparse.ArgumentParser:
    parser = subparsers.add_parser("benchmark", help="Run benchmark suite")
    parser.add_argument("--config", type=str, help="Path to YAML/JSON benchmark configuration")
    parser.add_argument("--modules", type=str, default="all", help="Modules to benchmark")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (cpu, cuda, etc.)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations")
    parser.add_argument("--iterations", type=int, default=50, help="Benchmark iterations")
    return parser


def handle_benchmark(args: argparse.Namespace) -> int:
    """Execute benchmark run with device environment configuration."""
    if getattr(args, "device", None) == "cpu":
        args.device = "cpu"
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    return 0
