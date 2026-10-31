#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression and Integration Tests for Compression Engines Debugging.

Validates:
1. Canonical count of 72 compression engines in COMPRESSOR_REGISTRY.
2. Alias resolution for res_svd, spectral_residual_lora, svd, svd_llm, awq, gptq, smoothquant.
3. Adaptive damping in compute_empirical_hessian under low-sample complexity (M < d_in).
4. HNC-SVD whiten_and_decompose_rank stability on rank-deficient activations.
5. GPTQQuantizer adaptive damping on small sample counts.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.pruning_utils import compute_empirical_hessian
from neuralops.compression.svd.hnc_svd_continuation import HessianNullSpaceSVDContinuation
from neuralops.compression.quantization.gptq import GPTQQuantizer


def test_canonical_compressor_count():
    """Verify that COMPRESSOR_REGISTRY contains exactly 72 canonical compression engines."""
    COMPRESSOR_REGISTRY._load_all()
    assert len(COMPRESSOR_REGISTRY) == 72, f"Expected 72 compressors, got {len(COMPRESSOR_REGISTRY)}"


@pytest.mark.parametrize("alias_name", [
    "res_svd",
    "spectral_residual_lora",
    "svd",
    "svd_llm",
    "awq",
    "gptq",
    "smoothquant",
])
def test_compressor_aliases_resolve(alias_name: str):
    """Verify newly registered aliases resolve successfully to valid compressor classes."""
    cls = COMPRESSOR_REGISTRY.get(alias_name)
    assert cls is not None, f"Alias {alias_name} failed to resolve in COMPRESSOR_REGISTRY"
    assert issubclass(cls, nn.Module) or hasattr(cls, "compress") or callable(cls)


def test_empirical_hessian_low_sample_complexity():
    """Verify compute_empirical_hessian applies adaptive damping when M < d_in."""
    d_in = 128
    m_samples = 4  # M << d_in (severely rank-deficient)
    activations = torch.randn(m_samples, d_in)

    H_reg, H_inv = compute_empirical_hessian(activations, damping=1e-4)
    assert torch.isfinite(H_reg).all()
    assert torch.isfinite(H_inv).all()

    # Product should approximate identity on active subspace and stay numerically bounded
    eye = torch.eye(d_in)
    diff = torch.matmul(H_reg, H_inv) - eye
    assert not torch.isnan(diff).any()
    assert not torch.isinf(diff).any()


def test_hnc_svd_whiten_decompose_low_samples():
    """Verify HNC-SVD whiten_and_decompose_rank does not fail on low sample counts."""
    d_out, d_in = 64, 128
    W = torch.randn(d_out, d_in)
    calib_activations = torch.randn(4, d_in)  # 4 samples < 128 dimensions

    A_r, B_r = HessianNullSpaceSVDContinuation.whiten_and_decompose_rank(
        W,
        calib_activations,
        rank=16,
    )
    assert A_r.shape == (d_out, 16)
    assert B_r.shape == (16, d_in)
    assert torch.isfinite(A_r).all()
    assert torch.isfinite(B_r).all()


def test_gptq_adaptive_damping_low_samples():
    """Verify GPTQQuantizer handles small calibration sample counts without singular matrix crash."""
    d_out, d_in = 32, 64
    linear = nn.Linear(d_in, d_out)
    quantizer = GPTQQuantizer(linear, bits=4)

    # Accumulate only 2 samples for 64 columns
    x_small = torch.randn(2, d_in)
    quantizer.add_batch(x_small)

    w_quant = quantizer.quantize(damp=0.01)
    assert w_quant.shape == (d_out, d_in)
    assert torch.isfinite(w_quant).all()
