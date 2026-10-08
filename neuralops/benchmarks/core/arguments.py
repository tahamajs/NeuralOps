"""Shared argument parser for all neuralops benchmarks."""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
import torch

LOG = logging.getLogger(__name__)


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments common to all benchmark runners."""
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        type=Path,
        default=Path("results"),
        help="Root directory for saving benchmark artifacts and results.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility across torch, numpy, and python.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Execution device target ('cuda', 'cpu', 'mps').",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "float16", "bfloat16"],
        help="Compute and tensor storage precision.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable detailed debug logs and progress output.",
    )
    parser.add_argument(
        "--dry-run",
        "--dry_run",
        action="store_true",
        help="Run validation checks and setup without executing full benchmarks.",
    )


def add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add model loading and architecture configuration arguments."""
    parser.add_argument(
        "--model",
        "--model-name",
        "--model_name",
        dest="model",
        type=str,
        required=True,
        help="HuggingFace model identifier or local directory path.",
    )
    parser.add_argument(
        "--model-path",
        "--model_path",
        dest="model_path",
        type=Path,
        default=None,
        help="Optional local path overriding default weights repository.",
    )
    parser.add_argument(
        "--model-family",
        "--model_family",
        type=str,
        default=None,
        help="Explicit model family architecture hint ('bert', 'roberta', 'gpt2', 'llama', 'qwen').",
    )


def add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """Add dataset specification, batching, and sequence length arguments."""
    parser.add_argument(
        "--dataset",
        "--dataset-name",
        "--dataset_name",
        dest="dataset",
        type=str,
        default="sst2",
        help="Evaluation dataset name (e.g. 'sst2', 'mrpc', 'qnli', 'wikitext2').",
    )
    parser.add_argument(
        "--dataset-dir",
        "--dataset_dir",
        type=Path,
        default=None,
        help="Local cached dataset directory path.",
    )
    parser.add_argument(
        "--seq-len",
        "--max-length",
        "--seq_len",
        "--max_length",
        dest="seq_len",
        type=int,
        default=128,
        help="Maximum input token sequence length.",
    )
    parser.add_argument(
        "--batch-size",
        "--batch_size",
        dest="batch_size",
        type=int,
        default=32,
        help="Evaluation and training mini-batch size.",
    )


def add_compression_args(parser: argparse.ArgumentParser) -> None:
    """Add compression algorithms, ranks, and adaptation parameters."""
    parser.add_argument(
        "--modules",
        "--module",
        dest="modules",
        nargs="+",
        default=["all"],
        help="One or more target attention modules or compression engines to evaluate.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=64,
        help="SVD projection or low-rank bottleneck dimension.",
    )
    parser.add_argument(
        "--recovery-steps",
        "--recovery_steps",
        type=int,
        default=50,
        help="Number of post-compression fine-tuning or recovery gradient steps.",
    )
    parser.add_argument(
        "--learning-rate",
        "--lr",
        "--learning_rate",
        dest="learning_rate",
        type=float,
        default=5e-5,
        help="Optimizer learning rate for post-compression fine-tuning.",
    )
    parser.add_argument(
        "--lora-rank",
        "--lora_rank",
        type=int,
        default=16,
        help="LoRA rank adaptation dimension if fine-tuning with adapters.",
    )
    parser.add_argument(
        "--acceptable-drop",
        "--acceptable_drop",
        type=float,
        default=5.0,
        help="Maximum allowed drop in percentage points relative to baseline accuracy.",
    )


__all__ = [
    "add_common_args",
    "add_model_args",
    "add_dataset_args",
    "add_compression_args",
    "LOG",
]
