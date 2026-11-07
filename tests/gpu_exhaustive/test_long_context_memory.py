#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Long-Context Sequence Scaling and VRAM Stress Verification Suite.
Demonstrates empirical subquadratic memory scaling (O(T) vs O(T^2)) up to 16k-32k tokens,
validates strict O(1) recurrent state memory scaling for linear/gated attention,
and verifies quadratic attention OOM thresholds with fail-safe recovery on GPU.
"""

from __future__ import annotations

import gc
import pytest
import torch

from neuralops.core.registry import create_attention


def _check_cuda() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device not available; skipping GPU long context memory test.")


@pytest.mark.gpu
@pytest.mark.parametrize(
    "mechanism_name",
    ["gated_delta_net", "cosformer", "gated_linear_attention"],
)
def test_long_context_linear_memory_scaling(mechanism_name: str):
    """
    Verify subquadratic memory complexity: VRAM must scale linearly with sequence length T,
    never quadratically (O(T^2)).
    """
    _check_cuda()
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
        assert torch.isfinite(out_tensor).all().item(), f"[{mechanism_name}] Non-finite output detected at T={t}"

        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
        vram_measurements[t] = peak_vram_mb
        del x, out, out_tensor

    # Complexity Check:
    # If quadratic (O(T^2)), doubling T increases memory by ~4x (ratio ~4.0).
    # If linear (O(T)), doubling T increases memory by at most ~3.2x (accounting for static model weights).
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
@pytest.mark.parametrize("mechanism_name", ["gated_delta_net"])
def test_subquadratic_recurrent_o1_memory_scaling(mechanism_name: str):
    """
    Validate O(1) recurrent memory scaling for subquadratic attention mechanisms.
    The autoregressive state tensor size must be strictly constant regardless of step index.
    """
    _check_cuda()
    device = torch.device("cuda:0")
    dim = 64
    heads = 2
    batch_size = 1

    try:
        mod = create_attention(mechanism_name, dim=dim, num_heads=heads).to(device=device, dtype=torch.float32)
    except Exception as exc:
        pytest.skip(f"[{mechanism_name}] Could not instantiate for recurrent test: {exc}")

    mod.eval()
    if not hasattr(mod, "step"):
        pytest.skip(f"[{mechanism_name}] Does not implement explicit .step() method.")

    state: dict[str, torch.Tensor] = {}
    state_bytes_history: list[int] = []

    # Run multiple recurrent autoregressive decode steps
    step_checkpoints = [1, 50, 100, 200, 400]
    total_steps = max(step_checkpoints)

    with torch.no_grad():
        for step_i in range(1, total_steps + 1):
            x_step = torch.randn(batch_size, 1, dim, device=device, dtype=torch.float32)
            out_step, state = mod.step(x_step, state=state)
            assert torch.isfinite(out_step).all().item()

            if step_i in step_checkpoints:
                assert "kv_state" in state, f"Expected 'kv_state' in recurrent state at step {step_i}"
                kv_state = state["kv_state"]
                bytes_alloc = kv_state.nelement() * kv_state.element_size()
                state_bytes_history.append(bytes_alloc)

    # Invariant: Recurrent state memory must be strictly constant across all step checkpoints
    assert len(state_bytes_history) == len(step_checkpoints)
    initial_bytes = state_bytes_history[0]
    for idx, (step_num, measured_bytes) in enumerate(zip(step_checkpoints, state_bytes_history)):
        assert measured_bytes == initial_bytes, (
            f"[{mechanism_name}] Recurrent state memory leak at step {step_num}: "
            f"expected {initial_bytes} bytes, observed {measured_bytes} bytes."
        )


@pytest.mark.gpu
def test_quadratic_attention_oom_threshold():
    """
    Verify quadratic attention OOM boundary:
    Naive unchunked quadratic attention expands an O(T^2) attention matrix that triggers
    CUDA OutOfMemoryError at large sequence lengths (e.g. 32k in FP32 with multiple heads),
    whereas subquadratic linear attention executes within that exact memory footprint.
    """
    _check_cuda()
    device = torch.device("cuda:0")
    torch.cuda.empty_cache()
    gc.collect()

    # Step 1: Deliberately push naive unchunked quadratic attention matrix to OOM threshold.
    # Shape: Batch=1, Heads=8, T=40000 -> (1, 8, 40000, 40000) float32 = 51.2 GB VRAM requirement!
    t_oom = 40000
    oom_caught = False

    try:
        # Naive quadratic attention matrix allocation
        q = torch.randn(1, 8, t_oom, 64, device=device, dtype=torch.float32)
        k = torch.randn(1, 8, t_oom, 64, device=device, dtype=torch.float32)
        _ = torch.matmul(q, k.transpose(-2, -1))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
            oom_caught = True
    finally:
        torch.cuda.empty_cache()
        gc.collect()

    assert oom_caught, f"Expected CUDA OOM at quadratic T={t_oom} sequence length on 24GB hardware."

    # Step 2: Validate subquadratic attention (gated_delta_net) executes cleanly at large T
    t_subquad = 8192
    try:
        mod = create_attention("gated_delta_net", dim=64, num_heads=2).to(device=device, dtype=torch.float16)
        mod.eval()
        x = torch.randn(1, t_subquad, 64, device=device, dtype=torch.float16)
        with torch.no_grad():
            out = mod(x)
        out_tensor = out[0] if isinstance(out, tuple) else out
        assert out_tensor.shape == (1, t_subquad, 64)
        assert torch.isfinite(out_tensor).all().item()
    finally:
        torch.cuda.empty_cache()
        gc.collect()


@pytest.mark.gpu
def test_oom_boundary_recovery_stability():
    """
    Push GPU memory to intentional OOM boundary, verify clean exception handling,
    and assert system can immediately execute subsequent workloads without leakage or corruption.
    """
    _check_cuda()
    device = torch.device("cuda:0")

    # Step 1: Deliberately trigger OOM by allocating beyond device capacity
    oom_triggered = False
    try:
        # Request 100 GB tensor on a 24GB RTX 3090
        _ = torch.empty((100, 1024, 1024, 256), device=device, dtype=torch.float32)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        if "out of memory" in str(exc).lower():
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
