#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying Rule 2.1 (torch.amp) and Rule 10 (no hardcoded cuda:0).

Ensures dynamic device resolution across roofline analysis, evaluation metrics,
and standalone evaluation scripts.
"""

from __future__ import annotations

from unittest.mock import patch
import torch

from neuralops.analysis.roofline import get_device_roofline_specs, profile_roofline
from neuralops.evaluation.metrics import measure_peak_vram_mb
from neuralops.evaluation.standalone.eval_ultra_long_context_scaling import (
    run_ultra_long_context_scaling,
)
from neuralops.runtime.config import RuntimeConfig


def test_roofline_dynamic_device_resolution_cpu() -> None:
    """Verifies roofline analysis dynamically resolves CPU when CUDA is absent."""
    with patch("torch.cuda.is_available", return_value=False):
        name, sm, tflops, bw = get_device_roofline_specs(device=None)
        assert sm == "cpu"
        assert tflops == 2.0
        assert bw == 50.0

        report = profile_roofline(
            module_name="gated_delta_net",
            device=None,
            dtype="float32",
            warmup=1,
            iters=1,
        )
        assert report.device_name == "CPU Host"
        assert report.sm_version == "cpu"


def test_roofline_dynamic_device_resolution_cuda() -> None:
    """Verifies roofline analysis uses current_device when device is None."""
    with (
        patch("torch.cuda.is_available", return_value=True),
        patch("torch.cuda.current_device", return_value=1),
        patch("torch.cuda.get_device_properties") as mock_props,
    ):
        mock_props.return_value.major = 8
        mock_props.return_value.minor = 9
        mock_props.return_value.name = "NVIDIA RTX 4090"
        mock_props.return_value.multi_processor_count = 128

        name, sm, tflops, bw = get_device_roofline_specs(device=None)
        assert name == "NVIDIA RTX 4090"
        assert sm == "SM 8.9"
        mock_props.assert_called_with(torch.device("cuda:1"))


def test_measure_peak_vram_dynamic_device() -> None:
    """Verifies measure_peak_vram_mb falls back to CPU cleanly when device=None."""
    with patch("torch.cuda.is_available", return_value=False):
        res, vram = measure_peak_vram_mb(lambda: 42, device=None)
        assert res == 42
        assert vram["peak_allocated_mb"] == 0.0
        assert vram["peak_reserved_mb"] == 0.0


def test_runtime_config_resolve_device() -> None:
    """Verifies RuntimeConfig.resolve_device obeys auto detection."""
    cfg = RuntimeConfig(device="auto")
    with patch("torch.cuda.is_available", return_value=False):
        dev = cfg.resolve_device()
        assert dev.type in ("mps", "cpu")


def test_ultra_long_context_scaling_default_device() -> None:
    """Verifies ultra long context scaling default device resolution."""
    with (
        patch("torch.cuda.is_available", return_value=False),
        patch("torch.cuda.current_device", return_value=0),
        patch("neuralops.evaluation.standalone.eval_ultra_long_context_scaling.AutoTokenizer"),
        patch("neuralops.evaluation.standalone.eval_ultra_long_context_scaling.build_model_for_context") as mock_build,
        patch("builtins.open"),
        patch("json.dump"),
    ):
        mock_model = patch("torch.nn.Module").start()
        mock_model.return_value = torch.zeros(1, 128, 10)
        mock_build.return_value = mock_model
        # When device_name is None on non-CUDA system, it defaults to CPU
        results = run_ultra_long_context_scaling(device_name=None, seq_lengths=[128], allow_cpu=True)
        assert results is not None
        mock_build.assert_called()
        call_args = mock_build.call_args_list[0][0]
        # Target device passed is cpu
        assert call_args[1] == torch.device("cpu")
