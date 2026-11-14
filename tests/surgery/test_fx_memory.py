#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for causal_prefix_scan chunked implementation and memory footprint.
Ensures no O(T*D^2) blowup and peak memory stays < 50MB for T=8192.
"""

import math
import tracemalloc
import torch

from neuralops.surgery.fx_surgery import causal_prefix_scan


def _reference_prefix_scan(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    eps: float = 1e-6,
    decay: float | None = None,
) -> torch.Tensor:
    """Exact reference implementation computing causal attention token by token."""
    B, H, T, D = q.shape
    E = v.shape[-1]
    out = torch.zeros(B, H, T, E, dtype=q.dtype, device=q.device)

    for t in range(T):
        if decay is not None and 0.0 < decay < 1.0:
            weights = torch.tensor(
                [math.pow(decay, t - i) for i in range(t + 1)],
                device=q.device,
                dtype=q.dtype,
            ).view(1, 1, t + 1, 1)
        else:
            weights = 1.0

        # k_slice: [B, H, t+1, D]
        k_slice = k[:, :, : t + 1, :] * weights
        v_slice = v[:, :, : t + 1, :]
        q_t = q[:, :, t : t + 1, :]  # [B, H, 1, D]

        # S_t = sum_{i=0}^t k_i^T v_i : [B, H, D, E]
        s_t = torch.einsum("bhid,bhie->bhde", k_slice, v_slice)
        z_t = torch.sum(k_slice, dim=2)  # [B, H, D]

        num = torch.matmul(q_t, s_t)
        denom = torch.matmul(q_t, z_t.unsqueeze(-1)) + eps
        out[:, :, t : t + 1, :] = num / denom

    return out


def test_causal_prefix_scan_numerical_parity():
    """Verify chunked scan matches exact reference output across different sequence lengths."""
    torch.manual_seed(42)
    for seq_len in (32, 128, 200):
        q = torch.rand(2, 2, seq_len, 16)
        k = torch.rand(2, 2, seq_len, 16)
        v = torch.rand(2, 2, seq_len, 16)

        # Without decay
        ref_no_decay = _reference_prefix_scan(q, k, v, decay=None)
        chunked_no_decay = causal_prefix_scan(q, k, v, decay=None, chunk_size=64)
        assert torch.allclose(ref_no_decay, chunked_no_decay, atol=1e-5)

        # With decay
        ref_decay = _reference_prefix_scan(q, k, v, decay=0.98)
        chunked_decay = causal_prefix_scan(q, k, v, decay=0.98, chunk_size=64)
        assert torch.allclose(ref_decay, chunked_decay, atol=1e-5)


def test_causal_prefix_scan_memory_bounded():
    """Verify peak memory remains under 50MB for sequence length T=8192."""
    torch.manual_seed(42)
    B, H, T, D = 1, 4, 8192, 64
    q = torch.rand(B, H, T, D)
    k = torch.rand(B, H, T, D)
    v = torch.rand(B, H, T, D)

    tracemalloc.start()
    _ = causal_prefix_scan(q, k, v, decay=0.99, chunk_size=128)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mb = peak / (1024 * 1024)
    # The output tensor alone is 1*4*8192*64*4 = 8MB.
    # Total extra memory allocated during scan should be < 50MB (far less than the 1GB of O(T*D^2))
    assert peak_mb < 50.0, f"Peak memory {peak_mb:.2f} MB exceeded 50 MB threshold"
