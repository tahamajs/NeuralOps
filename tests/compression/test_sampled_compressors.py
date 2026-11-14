# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Sampled Compressors Verification Suite (Wave 1 Audit).

Tests 15 representative compression algorithms applied to nn.Linear(64, 64):
- direct_svd, asvd, aasvd, adasvd, slicegpt, sliced_svd, a3, res_svd,
  saes_svd, nmf, rpca, cur, kronecker, kmeans, huffman.

Verifies:
1. Successful compression without crashes on valid linear layers.
2. Output shape preservation on forward pass (B, 64) -> (B, 64).
3. Numerical reconstruction validity (finite error, no NaNs or Infs).
4. Compression statistics reporting.
"""

from __future__ import annotations

import math
from typing import Any, Callable

import pytest
import torch
import torch.nn as nn

from neuralops.compression.low_rank.a3_compressor import A3Compressor
from neuralops.compression.low_rank.cur_compressor import CURCompressor
from neuralops.compression.low_rank.nmf_compressor import NMFCompressor
from neuralops.compression.low_rank.rpca_compressor import RPCACompressor
from neuralops.compression.quantization.huffman_coder import HuffmanCompressor
from neuralops.compression.quantization.kmeans_quantize import KMeansCompressor
from neuralops.compression.quantization.slicegpt import PCAActivationAwareSVDCompressor
from neuralops.compression.quantization.slicegpt_pca import SliceGPTPCACompressor
from neuralops.compression.svd.aa_svd_compressor import AASVDCompressor
from neuralops.compression.svd.adasvd_compressor import AdaSVDCompressor
from neuralops.compression.svd.asvd_compressor import ASVDCompressor
from neuralops.compression.svd.direct_svd import DirectSVDCompressor
from neuralops.compression.svd.residual_svd_compressor import ResSVDCompressor
from neuralops.compression.svd.saes_svd_v2 import SAESSVDCompressorV2
from neuralops.compression.tensor_decomp.kronecker_compressor import KroneckerCompressor


class SingleLinearModel(nn.Module):
    """Simple container wrapping an nn.Linear(64, 64)."""

    def __init__(self, in_features: int = 64, out_features: int = 64, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(in_features, out_features, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


COMPRESSOR_FACTORIES: dict[str, Callable[[nn.Module], Any]] = {
    "direct_svd": lambda m: DirectSVDCompressor(
        m, rank=16, skip_attention_qkv=False, target_attention_only=False, max_error=1.0
    ),
    "asvd": lambda m: ASVDCompressor(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "aasvd": lambda m: AASVDCompressor(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "adasvd": lambda m: AdaSVDCompressor(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "slicegpt": lambda m: PCAActivationAwareSVDCompressor(
        m, sparsity=0.5, rank=16, target_attention_only=False, max_error=1.0
    ),
    "sliced_svd": lambda m: SliceGPTPCACompressor(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "a3": lambda m: A3Compressor(
        m, rank=16, target_attention_only=False, recalibrate_ln=False, max_error=1.0
    ),
    "res_svd": lambda m: ResSVDCompressor(
        m, base_rank=16, residual_rank=4
    ),
    "saes_svd": lambda m: SAESSVDCompressorV2(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "nmf": lambda m: NMFCompressor(
        m, rank=16, max_iter=20, target_attention_only=False, max_error=1.0
    ),
    "rpca": lambda m: RPCACompressor(
        m, rank=16, max_iter=5, target_attention_only=False, max_error=1.0
    ),
    "cur": lambda m: CURCompressor(
        m, rank=16, target_attention_only=False, max_error=1.0
    ),
    "kronecker": lambda m: KroneckerCompressor(
        m, rank=16, target_attention_only=False, skip_attention_qkv=False, max_error=1.0
    ),
    "kmeans": lambda m: KMeansCompressor(
        m, n_clusters=16, target_attention_only=False, skip_attention_qkv=False, max_error=1.0
    ),
    "huffman": lambda m: HuffmanCompressor(
        m, quantize_bits=4, target_attention_only=False, max_error=1.0
    ),
}


@pytest.mark.parametrize("compressor_name", list(COMPRESSOR_FACTORIES.keys()))
def test_sampled_compressor_shapes_and_reconstruction(compressor_name: str) -> None:
    """Verify that each compressor compresses nn.Linear(64, 64) and preserves shape and numerical bounds."""
    torch.manual_seed(42)
    model = SingleLinearModel(64, 64, bias=True)

    batch_size = 2
    x = torch.randn(batch_size, 64)
    orig_output = model(x).detach()

    factory = COMPRESSOR_FACTORIES[compressor_name]
    compressor = factory(model)

    result = compressor.compress()
    compressed_model = result[0] if isinstance(result, tuple) else result
    stats = result[1] if isinstance(result, tuple) and len(result) > 1 else {}

    assert isinstance(compressed_model, nn.Module), f"{compressor_name} did not return an nn.Module"
    assert isinstance(stats, dict), f"{compressor_name} stats should be a dictionary"

    comp_output = compressed_model(x)
    assert comp_output.shape == (batch_size, 64), (
        f"{compressor_name} output shape mismatch: expected {(batch_size, 64)}, got {tuple(comp_output.shape)}"
    )

    # Numerical reconstruction checks
    assert not torch.isnan(comp_output).any(), f"{compressor_name} produced NaNs in forward pass"
    assert not torch.isinf(comp_output).any(), f"{compressor_name} produced Infs in forward pass"

    norm_orig = torch.norm(orig_output).item()
    norm_diff = torch.norm(orig_output - comp_output).item()
    assert norm_orig > 0, "Original output norm must be positive"
    rel_error = norm_diff / norm_orig

    assert not math.isnan(rel_error), f"{compressor_name} relative error is NaN"
    assert not math.isinf(rel_error), f"{compressor_name} relative error is Inf"
    # Even lossy compression of random weights should have bounded relative error
    assert 0.0 <= rel_error <= 2.0, f"{compressor_name} excessive relative error: {rel_error}"


def test_compressor_without_bias() -> None:
    """Verify that compressors handle nn.Linear without bias."""
    torch.manual_seed(42)
    model = SingleLinearModel(64, 64, bias=False)
    x = torch.randn(2, 64)

    compressor = DirectSVDCompressor(
        model, rank=16, skip_attention_qkv=False, target_attention_only=False, max_error=1.0
    )
    comp_model, stats = compressor.compress()
    out = comp_model(x)
    assert out.shape == (2, 64)
    assert not torch.isnan(out).any()
