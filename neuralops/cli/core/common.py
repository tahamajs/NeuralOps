#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Shared argument definitions and option builders for CLI parsers."""

from __future__ import annotations

import argparse


def add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add standard global CLI options (verbosity, formatting, output)."""
    parser.add_argument("--json", action="store_true", default=None, help="Format stdout as structured JSON")
    parser.add_argument("--quiet", "-q", action="store_true", default=None, help="Suppress non-essential progress and headers")
    parser.add_argument("--verbose", action="store_true", default=None, help="Print verbose execution traces and telemetry")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON/YAML configuration file")
    parser.add_argument("--device", type=str, default=None, help="Target device (cpu, cuda, cuda:0, mps, auto)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results and artifacts")
    parser.add_argument("--run-id", type=str, default=None, help="Unique identifier for the benchmark run")
    parser.add_argument("--overwrite", action="store_true", default=None, help="Overwrite existing output directory")


def add_benchmark_options(parser: argparse.ArgumentParser) -> None:
    """Add benchmark-specific execution and grid options with sentinel defaults."""
    parser.add_argument("--module", type=str, default=None, help="Single module name")
    parser.add_argument("--modules", type=str, default=None, help="Comma-separated operation keys, e.g. cuda:act_scale_gemm")
    parser.add_argument("--category", choices=["attention", "compressor", "cuda", "triton", "roofline", "quantization", "moe", "serving", "rope", "sampling", "ssm", "varlen", "sweep", "all"], default=None, help="Benchmark category")
    parser.add_argument("--family", type=str, default=None, help="Module family filter")
    parser.add_argument("--models", type=str, default=None, help="Comma-separated model names or HuggingFace IDs")
    parser.add_argument("--model", type=str, default=None, help="Single target model name")
    parser.add_argument("--all-models", action="store_true", default=None, help="Run against all 10 canonical model architectures")
    parser.add_argument("--datasets", type=str, default=None, help="Comma-separated dataset names")
    parser.add_argument("--dataset", type=str, default=None, help="Single dataset name")
    parser.add_argument("--all-datasets", action="store_true", default=None, help="Run across all 22 benchmark datasets")
    parser.add_argument("--ranks", type=str, default=None, help="Comma-separated rank integers")
    parser.add_argument("--rank", type=int, default=None, help="Single compression/low-rank dimension")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training/fine-tuning epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size for execution")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=None, help="Sequence length for benchmark runs")
    parser.add_argument("--dim", type=int, default=None, help="Hidden dimension")
    parser.add_argument("--heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--warmup", type=int, default=None, help="Number of warmup iterations")
    parser.add_argument("--iters", type=int, default=None, help="Number of timed benchmark iterations")
    parser.add_argument("--parallel", action="store_true", default=None, help="Enable parallel multi-GPU worker execution")
    parser.add_argument("--parallel-workers", type=int, default=None, help="Number of parallel workers")
    parser.add_argument("--dynamic", action="store_true", default=None, help="Enable dynamic shape benchmarking")
    parser.add_argument("--mixed-precision", choices=["none", "fp16", "bf16", "fp8"], default=None, help="Mixed precision mode")
    parser.add_argument("--dtype", type=str, default=None, help="Execution dtype (fp32, fp16, bf16)")
    parser.add_argument("--tolerance", type=float, default=None, help="Numerical tolerance for verification")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of modules evaluated")
    parser.add_argument("--yes", "-y", action="store_true", default=None, help="Auto-confirm execution for grids > 500 runs")
    parser.add_argument("--training-type", choices=["none", "scratch", "lora", "kd", "healing"], default=None, help="Training regime")
    parser.add_argument("--kd-teacher", type=str, default=None, help="Knowledge distillation teacher model")
    parser.add_argument("--kd-temperature", type=float, default=None, help="KD temperature")
    parser.add_argument("--kd-alpha", type=float, default=None, help="KD alpha weight")
    parser.add_argument("--recovery-steps", type=int, default=None, help="Post-surgery healing recovery steps")
    parser.add_argument("--grad-accum-steps", type=int, default=None, help="Gradient accumulation steps")
    parser.add_argument("--tracker", choices=["none", "local", "wandb", "mlflow", "tensorboard"], default=None, help="Experiment tracker")
    parser.add_argument("--project-name", type=str, default=None, help="Experiment tracking project name")
    parser.add_argument("--run-name", type=str, default=None, help="Experiment run name")
    parser.add_argument("--log-interval", type=int, default=None, help="Logging step interval")
    parser.add_argument("--distributed-backend", choices=["none", "ddp", "fsdp", "deepspeed"], default=None, help="Distributed backend")
    parser.add_argument("--no-compression", action="store_true", default=None, help="Run baseline uncompressed model only")
    parser.add_argument("--force-rerun", action="store_true", default=None, help="Force re-execution of previously completed runs")
