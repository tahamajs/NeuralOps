#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical serving latency benchmark backed by the shared result schema."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats
from neuralops.benchmarks.serving import run_benchmark


def run_serving_matrix(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Measure the local serving primitive and persist a typed result."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for serving, but CUDA is unavailable.")
    result_data = run_benchmark(
        device=config.device,
        prompt_tokens=config.seq_len,
        decode_tokens=max(1, config.iters),
    )
    result = BenchmarkResult(
        module={"name": result_data["module"], "type": "serving", "family": "inference"},
        model={"name": "synthetic_serving_probe"},
        dataset={"name": "synthetic", "task": "serving"},
        config={"device": result_data["device"], "prompt_tokens": config.seq_len, "decode_tokens": config.iters},
        metrics={
            "ttft_ms": float(result_data["ttft_ms"]),
            "itl_ms": float(result_data["itl_ms"]),
            "requests_per_second": float(result_data["requests_per_second"]),
        },
        status="success" if result_data["status"] == "PASS" else "FAIL_NUMERICS",
        error=None if result_data["status"] == "PASS" else "Serving output was non-finite",
    )
    save_all_formats(result, Path(config.output_dir))
    if result.status != "success":
        raise RuntimeError(result.error or "Serving benchmark failed")
    return [result]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args()
    config = BenchmarkConfig.from_file(args.config)
    if args.device:
        config.device = args.device
    run_serving_matrix(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
