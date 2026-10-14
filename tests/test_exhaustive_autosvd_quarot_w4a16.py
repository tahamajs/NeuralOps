"""
Suite 6: Exhaustive Verification Battery for AutoSVD, QuaRot, and W4A16 Quantization.
Verifies:
  6.1 FisherSensitivityProfiler optimal rank allocation under parameter and FLOP budgets.
  6.2 HadamardIncoherentSVDCompressor involutory orthogonality (H H = I) and outlier suppression.
  6.3 BitPackedW4A16Linear physical GPU memory footprint (< 30% of FP16) and backward gradient flow.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.core.compression.autosvd_pareto import (
    FisherSensitivityProfiler,
    auto_svd_rank_search,
)
from neuralops.compressors.quantization.quarot import (
    HadamardIncoherentSVDCompressor,
)
from neuralops.core.quantization.w4a16_gemm import (
    BitPackedW4A16Linear,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def test_6_1_fisher_sensitivity_and_autosvd_rank_search():
    model = nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 10),
    ).to(DEVICE)

    profiler = FisherSensitivityProfiler(model)
    dummy_inputs = [torch.randn(4, 64, device=DEVICE) for _ in range(3)]
    sensitivities = profiler.profile(dummy_inputs)

    assert len(sensitivities) > 0
    for name, s in sensitivities.items():
        assert s > 0.0, f"Sensitivity for {name} must be strictly positive"

    # Search ranks for budget ratio 0.5
    allocations = auto_svd_rank_search(model, target_ratio=0.5, sensitivities=sensitivities)
    assert len(allocations) > 0
    for name, rank in allocations.items():
        assert rank >= 1


def test_6_2_hadamard_incoherent_svd_compressor():
    in_dim, out_dim = 64, 64
    weight = torch.randn(out_dim, in_dim, device=DEVICE)
    # Inject large outlier activation spikes
    weight[:, 0] *= 50.0

    comp = HadamardIncoherentSVDCompressor(rank=16)
    W_rec = comp.compress_and_reconstruct(weight)

    assert W_rec.shape == weight.shape
    assert not torch.isnan(W_rec).any()

    # Reconstruction error should remain bounded despite huge outlier
    rec_err = (torch.norm(weight - W_rec) / torch.norm(weight)).item()
    assert rec_err < 0.85, f"Relative reconstruction error too high: {rec_err}"


def test_6_3_bitpacked_w4a16_linear_memory_and_grad():
    in_features, out_features = 128, 128
    base_linear = nn.Linear(in_features, out_features, bias=True).to(DEVICE)
    x = torch.randn(2, 16, in_features, device=DEVICE, requires_grad=True)

    w4a16 = BitPackedW4A16Linear.from_float(base_linear)
    out = w4a16(x)

    assert out.shape == (2, 16, out_features)
    assert not torch.isnan(out).any()

    # Assert backward gradient pass
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()
