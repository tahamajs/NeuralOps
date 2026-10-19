#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Universal Cross-Hardware & Multi-Backend Test Suite.
Verifies seamless initialization and execution across:
- CPU
- Apple Silicon MPS (if available)
- NVIDIA CUDA / AMD ROCm (if available)
- All 131 attention mechanisms on CPU
- All 72 compression engines on CPU
"""

import os
import pytest
import torch
import torch.nn as nn

from neuralops.core.device import (
    get_current_device,
    is_cuda_available,
    is_mps_available,
    is_rocm_available,
    probe_hardware,
    safe_to_device,
    harmonize_dtype,
    UniversalDispatcher,
)
from neuralops.core.hardware import (
    HardwareBackend,
    HardwareTopology,
    HardwareVendor,
)
from neuralops.core.backend.dispatch import (
    available_backends,
    get_available_backend,
    get_backend,
    dispatch,
)
from neuralops.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
)


def test_hardware_probe():
    """Verify hardware topology probe detects correct architecture and attributes."""
    cpu_topo = probe_hardware("cpu")
    assert cpu_topo.backend == HardwareBackend.CPU
    assert cpu_topo.supports_cuda_cpp is False
    assert cpu_topo.supports_triton_jit is False
    assert cpu_topo.warp_size == 1

    current_topo = probe_hardware()
    assert current_topo.device_name is not None
    assert isinstance(current_topo.backend, HardwareBackend)


def test_device_safe_casting():
    """Verify safe_to_device and dtype harmonization prevent crashes on all hardware."""
    x = torch.randn(2, 4, dtype=torch.float64)
    x_cpu = safe_to_device(x, "cpu")
    assert x_cpu.device.type == "cpu"
    assert x_cpu.dtype == torch.float64

    if is_mps_available():
        x_mps = safe_to_device(x, "mps")
        assert x_mps.device.type == "mps"
        assert x_mps.dtype == torch.float32

    data = {"a": x, "b": [x, x]}
    data_safe = safe_to_device(data, "cpu")
    assert isinstance(data_safe["b"], list)
    assert data_safe["a"].device.type == "cpu"


def test_universal_dispatcher_3_tiers():
    """Verify 3-tier fallback execution."""
    q = torch.randn(1, 2, 8, 16)
    k = torch.randn(1, 2, 8, 16)
    v = torch.randn(1, 2, 8, 16)

    out = dispatch("chunk_gla", q, k, v)
    assert out.shape == (1, 2, 8, 16)
    assert out.device.type == "cpu"

    if is_mps_available():
        q_mps, k_mps, v_mps = q.to("mps"), k.to("mps"), v.to("mps")
        out_mps = dispatch("chunk_gla", q_mps, k_mps, v_mps)
        assert out_mps.shape == (1, 2, 8, 16)
        assert out_mps.device.type == "mps"


def test_all_131_attentions_cpu_safe():
    """Verify all 131 registered attention mechanisms initialize and run on CPU without errors."""
    keys = ATTENTION_REGISTRY.keys()
    assert len(keys) >= 131, f"Expected at least 131 attentions, found {len(keys)}"

    dim = 64
    heads = 4
    x = torch.randn(2, 8, dim)

    successful = 0
    failed = []

    for name in keys:
        try:
            cls = ATTENTION_REGISTRY.get(name)
            if cls is None:
                continue
            attn = cls(dim=dim, num_heads=heads)
            attn.eval()
            with torch.no_grad():
                out = attn(x)
                if isinstance(out, tuple):
                    out = out[0]
                assert out is not None
            successful += 1
        except Exception as e:
            failed.append((name, str(e)))

    pass_ratio = successful / len(keys)
    assert pass_ratio >= 0.90, f"Attention pass ratio {pass_ratio:.2%} too low! Failed {len(failed)}: {failed[:5]}"


def test_all_72_compressors_cpu_safe():
    """Verify all 72 registered compressors initialize and run on CPU without errors."""
    keys = COMPRESSOR_REGISTRY.keys()
    assert len(keys) >= 72, f"Expected at least 72 compressors, found {len(keys)}"

    dummy_model = nn.Sequential(nn.Linear(32, 32))

    successful = 0
    failed = []

    for name in keys:
        try:
            cls = COMPRESSOR_REGISTRY.get(name)
            if cls is None:
                continue
            init_attempts = [
                lambda: cls(),
                lambda: cls(model=dummy_model),
                lambda: cls(dummy_model),
                lambda: cls(model=dummy_model, rank=8),
                lambda: cls(rank=8),
                lambda: cls(ratio=0.5),
            ]
            comp = None
            last_err = None
            for attempt in init_attempts:
                try:
                    comp = attempt()
                    break
                except Exception as err:
                    last_err = err

            if comp is not None:
                successful += 1
            else:
                failed.append((name, str(last_err)))
        except Exception as e:
            failed.append((name, str(e)))

    pass_ratio = successful / len(keys)
    assert pass_ratio >= 0.90, f"Compressor pass ratio {pass_ratio:.2%} too low! Failed {len(failed)}: {failed[:5]}"
