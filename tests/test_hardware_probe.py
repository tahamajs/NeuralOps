#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for hardware topology probing and capability detection (F-45)."""

from __future__ import annotations

from unittest.mock import patch
import torch

from neuralops.core.hardware import HardwareTopology, probe_hardware


def test_probe_hardware_cpu_bf16_is_detected_not_assumed() -> None:
    """Verify probe_hardware('cpu') detects bfloat16 capability rather than assuming True."""
    probe_hardware.cache_clear()
    topo: HardwareTopology = probe_hardware("cpu")
    assert isinstance(topo, HardwareTopology)
    assert isinstance(topo.supports_bf16, bool)

    # When CPU capability does not report AVX512_BF16 / AMX_BF16, supports_bf16 should be False
    with patch("neuralops.core.hardware._detect_cpu_bf16_support", return_value=False):
        probe_hardware.cache_clear()
        topo_no_bf16 = probe_hardware("cpu")
        assert topo_no_bf16.supports_bf16 is False
        assert torch.bfloat16 not in topo_no_bf16.supported_dtypes

    # When detection reports True, supports_bf16 should reflect it
    with patch("neuralops.core.hardware._detect_cpu_bf16_support", return_value=True):
        probe_hardware.cache_clear()
        topo_bf16 = probe_hardware("cpu")
        assert topo_bf16.supports_bf16 is True
        assert torch.bfloat16 in topo_bf16.supported_dtypes
    probe_hardware.cache_clear()


def test_probe_hardware_cuda_branch_guarded_when_unavailable() -> None:
    """Verify probe_hardware('cuda') is guarded when CUDA is unavailable and does not raise."""
    probe_hardware.cache_clear()
    with patch("torch.cuda.is_available", return_value=False):
        topo: HardwareTopology = probe_hardware("cuda")
        assert isinstance(topo, HardwareTopology)
        assert topo.supports_cuda_cpp is False
        assert topo.device_name == ""
        assert topo.compute_capability == "0.0"
        assert topo.supports_triton_jit is False
    probe_hardware.cache_clear()
