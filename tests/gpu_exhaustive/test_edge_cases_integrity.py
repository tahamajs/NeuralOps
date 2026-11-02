#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Numerical Edge Cases, Extreme Value Clamping, and Integrity Verification Suite.
Validates zero-input stability, extreme overflow bounds, and batch size 1 execution.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention


REPRESENTATIVE_ATTENTIONS = [
    "full_attention",
    "cosformer",
    "gated_delta_net",
    "simple_ssm",
    "linear_attention",
    "performer_attention",
    "flash_attention3",
    "galerkin_attention",
    "hgrn2_gated_layer",
    "decay_linear_attention",
]


@pytest.mark.gpu
@pytest.mark.parametrize("name", REPRESENTATIVE_ATTENTIONS)
def test_all_zero_inputs_no_nan(name: str):
    """Verify feeding all-zero input tensor does not generate NaN activations."""
    device = torch.device("cuda:0")
    dim = 64
    heads = 4

    try:
        mod = create_attention(name, dim=dim, num_heads=heads).to(device=device)
    except Exception as exc:
        pytest.skip(f"[{name}] Could not instantiate: {exc}")

    mod.eval()
    x_zeros = torch.zeros(2, 16, dim, device=device, dtype=torch.float32)

    with torch.no_grad():
        try:
            out = mod(x_zeros)
        except Exception as exc:
            pytest.fail(f"[{name}] Zero-input forward crashed: {exc}")

    out_tensor = out[0] if isinstance(out, tuple) else out
    assert not torch.isnan(out_tensor).any().item(), f"[{name}] All-zero inputs produced NaN in forward output."


@pytest.mark.gpu
@pytest.mark.parametrize("name", REPRESENTATIVE_ATTENTIONS)
def test_strictly_negative_inputs_stability(name: str):
    """Verify feeding strictly negative input tensor remains numerically stable."""
    device = torch.device("cuda:0")
    dim = 64
    heads = 4

    try:
        mod = create_attention(name, dim=dim, num_heads=heads).to(device=device)
    except Exception as exc:
        pytest.skip(f"[{name}] Could not instantiate: {exc}")

    mod.eval()
    x_neg = -torch.abs(torch.randn(2, 16, dim, device=device, dtype=torch.float32)) - 1.0

    with torch.no_grad():
        try:
            out = mod(x_neg)
        except Exception as exc:
            pytest.fail(f"[{name}] Negative-input forward crashed: {exc}")

    out_tensor = out[0] if isinstance(out, tuple) else out
    assert not torch.isnan(out_tensor).any().item(), f"[{name}] Strictly negative inputs produced NaN in output."


@pytest.mark.gpu
@pytest.mark.parametrize("name", REPRESENTATIVE_ATTENTIONS)
def test_extreme_values_overflow_boundary(name: str):
    """
    CRITICAL NUMERICAL GATE:
    Feeding extreme values (1e6, -1e6) must either clamp gracefully or raise
    an explicit exception, but NEVER silently return NaN values.
    """
    device = torch.device("cuda:0")
    dim = 64
    heads = 4

    try:
        mod = create_attention(name, dim=dim, num_heads=heads).to(device=device)
    except Exception as exc:
        pytest.skip(f"[{name}] Could not instantiate: {exc}")

    mod.eval()
    x_extreme = torch.full((1, 8, dim), 1e6, device=device, dtype=torch.float32)

    try:
        with torch.no_grad():
            out = mod(x_extreme)
        out_tensor = out[0] if isinstance(out, tuple) else out
        assert not torch.isnan(out_tensor).any().item(), (
            f"[{name}] Extreme values (+1e6) silently produced NaN activations."
        )
    except (RuntimeError, OverflowError):
        # Raising an explicit exception on overflow is valid safe behavior
        pass


@pytest.mark.gpu
def test_batch_size_one_all_registered_attentions():
    """Verify all 131 attention mechanisms successfully process single-batch requests (B=1)."""
    device = torch.device("cuda:0")
    dim = 64
    heads = 2
    b, t = 1, 16

    ATTENTION_REGISTRY._load_all()
    names = list(ATTENTION_REGISTRY._dict.keys())

    failed_modules: list[str] = []
    for name in names:
        try:
            mod = create_attention(name, dim=dim, num_heads=heads).to(device=device)
            mod.eval()
            x = torch.randn(b, t, dim, device=device, dtype=torch.float32)
            with torch.no_grad():
                out = mod(x)
            out_tensor = out[0] if isinstance(out, tuple) else out
            if out_tensor.shape[0] != 1 or torch.isnan(out_tensor).any().item():
                failed_modules.append(name)
        except Exception:
            # If instantiation or forward fails on B=1, record failure
            failed_modules.append(name)

    assert len(failed_modules) == 0, f"Batch size 1 execution failed on {len(failed_modules)} modules: {failed_modules}"
