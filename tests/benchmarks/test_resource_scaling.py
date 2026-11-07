#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit and contract tests for resource scaling and fail-closed CUDA behavior."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.resource_scaling import main as resource_scaling_main
from neuralops.benchmarks.resource_scaling import run_resource_scaling


def test_resource_scaling_cpu_contract(tmp_path: Path):
    """Verify that resource scaling runs deterministically on CPU and writes results."""
    config = BenchmarkConfig(
        device="cpu",
        batch_sizes=[1],
        seq_lens=[32],
        precision="fp32",
        warmup=1,
        iters=2,
        output_dir=tmp_path,
    )
    results = run_resource_scaling(config)
    assert len(results) == 1
    res = results[0]
    assert res.status == "success"
    assert res.metrics["latency_ms"] > 0
    assert res.metrics["tokens_per_second"] > 0
    assert res.config["device"] == "cpu"
    assert (tmp_path / "results.jsonl").exists()


def test_resource_scaling_unsupported_device_fail_closed(tmp_path: Path):
    """Verify fail-closed behavior when CUDA is requested but unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        config_cuda = BenchmarkConfig(device="cuda", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="CUDA was requested for resource scaling, but CUDA is unavailable"):
            run_resource_scaling(config_cuda)

        config_cuda0 = BenchmarkConfig(device="cuda:0", output_dir=tmp_path)
        with pytest.raises(RuntimeError, match="CUDA was requested for resource scaling, but CUDA is unavailable"):
            run_resource_scaling(config_cuda0)


def test_resource_scaling_cli_exit_code_on_missing_backend(tmp_path: Path):
    """CLI must return exit code 3 (backend unavailable) when CUDA is requested without hardware."""
    with patch("torch.cuda.is_available", return_value=False):
        with patch.object(sys, "argv", ["resource_scaling", "--device", "cuda", "--output-dir", str(tmp_path)]):
            exit_code = resource_scaling_main()
            assert exit_code == 3


def test_eval_ultra_long_context_fail_closed():
    """Verify eval_ultra_long_context_scaling fail-closed contract when CUDA unavailable."""
    from neuralops.evaluation.standalone.eval_ultra_long_context_scaling import (
        main as eval_main,
        run_ultra_long_context_scaling,
    )

    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="requires CUDA"):
            run_ultra_long_context_scaling(device_name="cuda:0", allow_cpu=False)

        with patch.object(sys, "argv", ["eval_ultra_long_context", "--device", "cuda:0"]):
            exit_code = eval_main()
            assert exit_code == 3
