#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Command-line launcher for isolated, resource-intensive evaluation runs."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_EXPERIMENTS = {
    "comprehensive": ("master_comprehensive_evaluation", False),
    "concurrent-offloading": ("eval_concurrent_offloading_jitter", True),
    "continuous-stress": ("eval_continuous_500token_stress", True),
    "cot-jitter": ("eval_cot_jitter_128steps", True),
    "arithmetic-continuation": ("eval_arithmetic_continuation_256steps", True),
    "reasoning-scaling": ("eval_reasoning_and_cot_scaling", True),
    "reasoning-long-context": ("eval_reasoning_long_context", True),
    "speculative-decoding": ("eval_speculative_decoding", True),
    "speculative-sweep": ("eval_speculative_sweep_k1_k8", True),
    "ultra-long-context": ("eval_ultra_long_context_scaling", True),
}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neuralops-eval",
        description="Launch a standalone NeuralOps evaluation experiment in a subprocess.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("list", help="list available experiment names")

    run_parser = subparsers.add_parser("run", help="run one experiment")
    run_parser.add_argument("experiment", choices=sorted(_EXPERIMENTS))
    run_parser.add_argument(
        "--model-path",
        default=os.environ.get("NEURALOPS_MODEL_PATH", "gpt2"),
        help="Hugging Face model ID or local model directory (default: gpt2)",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("NEURALOPS_OUTPUT_DIR", "results/evaluation")),
        help="directory for generated JSON/Markdown reports",
    )
    run_parser.add_argument(
        "--device",
        default=None,
        help="device for the comprehensive experiment; other experiments require CUDA",
    )
    run_parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=None,
        help="positive sequence lengths for the comprehensive matrix",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Validate run settings and start the selected experiment in isolation."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "list":
        for name, (module, requires_cuda) in sorted(_EXPERIMENTS.items()):
            requirement = "CUDA required" if requires_cuda else "CPU or CUDA"
            print(f"{name:26} {requirement:14} neuralops.evaluation.standalone.{module}")
        return 0

    module_name, requires_cuda = _EXPERIMENTS[args.experiment]
    if args.device is not None and module_name != "master_comprehensive_evaluation":
        parser.error("--device is currently supported only by the comprehensive experiment")
    if args.context_lengths is not None:
        if module_name != "master_comprehensive_evaluation":
            parser.error("--context-lengths is supported only by the comprehensive experiment")
        if any(length <= 0 for length in args.context_lengths):
            parser.error("--context-lengths values must be positive integers")

    try:
        import torch
    except ImportError as exc:
        parser.error(f"PyTorch is required to run an experiment: {exc}")

    if requires_cuda and not torch.cuda.is_available():
        parser.error(
            f"{args.experiment} requires CUDA; the comprehensive experiment is the only "
            "entry that currently permits CPU execution"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["NEURALOPS_MODEL_PATH"] = args.model_path
    env["NEURALOPS_OUTPUT_DIR"] = str(args.output_dir.resolve())

    command = [sys.executable, "-m", f"neuralops.evaluation.standalone.{module_name}"]
    if args.device is not None:
        command.extend(["--device", args.device])
    if args.context_lengths is not None:
        command.extend(["--context-lengths", *(str(length) for length in args.context_lengths)])

    completed = subprocess.run(command, env=env, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
