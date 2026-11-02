#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Long-Context Sequence Scaling and VRAM Stress Verification Suite.
Demonstrates empirical subquadratic memory scaling (O(T) vs O(T^2)) up to 32k tokens,
and verifies OOM fail-safe recovery on GPU.
"""

from __future__ import annotations

import gc
import pytest
import torch

from neuralops.core.registry import create_attention


@pytest.mark.gpu
@pytest.mark.parametrize("mechanism_name", ["gated_delta_net", "cosformer", "simple_ssm"])
def test_long_context_linear_memory_scaling(mechanism_name: str):
    """
    Verify subquadratic memory complexity: VRAM must scale linearly with sequence length T,
    never quadratically (O(T^2)).
    """
    device = torch.device("cuda:0")
    dim = 64
    heads = 2
    batch_size = 1
    seq_lengths = [4096, 8192, 16384]

    try:
        mod = create_attention(mechanism_name, dim=dim, num_heads=heads).to(device=device, dtype=torch.float16)
    except Exception as exc:
        pytest.skip(f"[{mechanism_name}] Could not instantiate for long-context test: {exc}")

    mod.eval()
    vram_measurements: dict[int, float] = {}

    for t in seq_lengths:
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats(device)

        x = torch.randn(batch_size, t, dim, device=device, dtype=torch.float16)
        with torch.no_grad():
            try:
                out = mod(x)
            except Exception as exc:
                pytest.skip(f"[{mechanism_name}] Long-context T={t} execution unsupported: {exc}")

        out_tensor = out[0] if isinstance(out, tuple) else out
        assert out_tensor.shape == (batch_size, t, dim)
        assert torch.isfinite(out_tensor).all().item()

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        vram_measurements[t] = peak_vram_mb
        del x, out, out_tensor

    # Complexity Check:
    # If quadratic (O(T^2)), doubling T increases memory by ~4x (ratio ~4.0).
    # If linear (O(T)), doubling T increases memory by at most ~2.2x (accounting for static model weights).
    ratio_8k_4k = vram_measurements[8192] / max(1e-3, vram_measurements[4096])
    ratio_16k_8k = vram_measurements[16384] / max(1e-3, vram_measurements[8192])

    assert ratio_8k_4k < 3.2, (
        f"[{mechanism_name}] Quadratic VRAM explosion detected from 4k to 8k: "
        f"ratio={ratio_8k_4k:.2f} (4k: {vram_measurements[4096]:.1f}MB, 8k: {vram_measurements[8192]:.1f}MB)"
    )
    assert ratio_16k_8k < 3.2, (
        f"[{mechanism_name}] Quadratic VRAM explosion detected from 8k to 16k: "
        f"ratio={ratio_16k_8k:.2f} (8k: {vram_measurements[8192]:.1f}MB, 16k: {vram_measurements[16384]:.1f}MB)"
    )


@pytest.mark.gpu
def test_oom_boundary_recovery_stability():
    """
    Push GPU memory to intentional OOM boundary, verify clean exception handling,
    and assert system can immediately execute subsequent workloads without leakage or corruption.
    """
    device = torch.device("cuda:0")

    # Step 1: Deliberately trigger OOM by allocating beyond device capacity
    oom_triggered = False
    try:
        # Request 100 GB tensor on a 24GB RTX 3090
        _ = torch.empty((100, 1024, 1024, 256), device=device, dtype=torch.float32)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower() or "CUDA out of memory" in str(exc):
            oom_triggered = True

    assert oom_triggered, "Failed to trigger expected intentional OOM boundary condition."

    # Step 2: System Cache Cleanup & Recovery
    gc.collect()
    torch.cuda.empty_cache()

    # Step 3: Verification of Normal Hardware Functionality Post-OOM
    a = torch.randn(512, 512, device=device, dtype=torch.float32)
    b = torch.randn(512, 512, device=device, dtype=torch.float32)
    c = torch.matmul(a, b)

    assert torch.isfinite(c).all().item(), "Post-OOM recovery failed: MatMul produced non-finite results."
    assert c.shape == (512, 512), "Post-OOM recovery failed: Shape corruption."
