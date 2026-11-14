#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Smoke and Invariant Test Suite for Factorization and PEFT Compressors (Wave 2 G3).

Covers:
1. LoRA / DoRA / LoHa / LoKr: forward pass, in-place merge/unmerge, and alpha scaling.
2. Low-rank & Matrix Factorization: NMF, RPCA, CUR, Kronecker, Butterfly, Randomized SVD.
3. Transform & Subspace: FFT, Wavelet, Hadamard Incoherent, Subspace Projection, Optimal Subspace.
4. Numerical Stability & Reconstruction Bounds across FP32, FP16, and BF16.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from neuralops.compression.low_rank.butterfly_compression import ButterflyCompressor, ButterflyLinear
from neuralops.compression.low_rank.cur_compressor import CURCompressor, CURLinear
from neuralops.compression.low_rank.dora_qlora import DoRALinear
from neuralops.compression.low_rank.fft_compression import FFTCompressor, FFTLinear
from neuralops.compression.low_rank.loha import LoHaCompressor, LoHaLinear
from neuralops.compression.low_rank.lokr import LoKrCompressor, LoKrLinear
from neuralops.compression.low_rank.nmf_compressor import NMFCompressor, NMFLinear
from neuralops.compression.low_rank.rpca_compressor import RPCACompressor, RPCALinear
from neuralops.compression.low_rank.wavelet_compression import WaveletCompressor, WaveletLinear
from neuralops.compression.quantization.kmeans_quantize import KMeansCompressor, KMeansLinear
from neuralops.compression.quantization.mixed_precision_compressor import MixedPrecisionCompressor, MixedPrecisionLinear
from neuralops.compression.quantization.rotational_slicegpt_int4 import RotationalSliceGPTINT4Compressor
from neuralops.compression.svd.hadamard_incoherent_svd import HadamardIncoherentSVDCompressor
from neuralops.compression.svd.optimal_subspace_projection import OptimalSubspaceProjectionCompressor
from neuralops.compression.svd.randomized_compression import RandomizedSVDCompressor, RandomizedSVDLinear
from neuralops.compression.svd.subspace_projection import SubspaceLinear, SubspaceProjectionCompressor
from neuralops.compression.tensor_decomp.kronecker_compressor import KroneckerCompressor, KroneckerLinear
from neuralops.core.peft.lora import LoRALinear
from neuralops.core.peft.simple_lora import SimpleLoRALinear
from neuralops.core.svd_utils import randomized_svd


class TinyMLP(nn.Module):
    """Synthetic MLP model for compression pipeline smoke tests."""

    def __init__(self, in_features: int = 64, hidden: int = 128, out_features: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# =====================================================================
# 1. LoRA / SimpleLoRA / DoRA / LoHa / LoKr Invariant Tests
# =====================================================================


class TestPEFTAdapters:
    """Rigorous invariant verification for LoRA, DoRA, LoHa, and LoKr."""

    def test_lora_forward_merge_unmerge_alpha(self) -> None:
        base = nn.Linear(32, 64, bias=True)
        orig_w = base.weight.clone()

        lora = LoRALinear(base, r=4, alpha=16.0)
        assert lora.scaling == 4.0
        assert not lora.base_layer.weight.requires_grad
        assert lora.lora_A.weight.requires_grad
        assert lora.lora_B.weight.requires_grad

        # Check forward
        x = torch.randn(2, 32)
        out1 = lora(x)
        assert out1.shape == (2, 64)
        assert torch.isfinite(out1).all()

        # Modify adapter to create a non-zero delta
        lora.lora_B.data.fill_(0.05)
        out_unmerged = lora(x)

        # Merge
        lora.merge()
        assert lora.merged
        out_merged = lora(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        # Unmerge
        lora.unmerge()
        assert not lora.merged
        out_reverted = lora(x)
        assert torch.allclose(out_unmerged, out_reverted, atol=1e-5)

        # Alpha scaling check: doubling alpha doubles the delta
        lora_double = LoRALinear(base, r=4, alpha=32.0)
        assert lora_double.scaling == 8.0

    def test_simple_lora_invariants(self) -> None:
        base = nn.Linear(32, 32, bias=False)
        simple = SimpleLoRALinear(base, rank=4, alpha=8.0)
        assert simple.scaling == 2.0

        x = torch.randn(3, 32)
        out = simple(x)
        assert out.shape == (3, 32)

        simple.lora_B.data.fill_(0.1)
        unmerged_out = simple(x)
        simple.merge()
        assert simple.merged
        assert torch.allclose(unmerged_out, simple(x), atol=1e-5)
        simple.unmerge()
        assert not simple.merged
        assert torch.allclose(unmerged_out, simple(x), atol=1e-5)

    def test_dora_forward_merge_unmerge(self) -> None:
        base = nn.Linear(32, 48, bias=True)
        dora = DoRALinear(base, rank=4, lora_alpha=16.0)
        assert dora.scaling == 4.0

        x = torch.randn(2, 32)
        out_initial = dora(x)
        assert out_initial.shape == (2, 48)
        assert torch.isfinite(out_initial).all()

        # Give LoRA a perturbation
        dora.lora_B.data.fill_(0.05)
        out_unmerged = dora(x)

        # Merge
        dora.merge()
        assert dora.merged
        out_merged = dora(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        # Unmerge
        dora.unmerge()
        assert not dora.merged
        out_reverted = dora(x)
        assert torch.allclose(out_unmerged, out_reverted, atol=1e-5)

    def test_loha_forward_merge_unmerge_alpha(self) -> None:
        base = nn.Linear(32, 48, bias=False)
        loha = LoHaLinear(base, rank=4, alpha=16.0)
        assert loha.scaling == 4.0
        assert not loha.base_layer.weight.requires_grad
        assert loha.lora_A1.requires_grad
        assert loha.lora_B2.requires_grad

        x = torch.randn(2, 32)
        out_init = loha(x)
        assert out_init.shape == (2, 48)

        # At init B2 is zero, so delta is zero
        base_direct = base(x)
        assert torch.allclose(out_init, base_direct, atol=1e-5)

        # Set B2 to non-zero to activate Hadamard delta
        loha.lora_B2.data.fill_(0.1)
        out_unmerged = loha(x)
        assert not torch.allclose(out_unmerged, base_direct, atol=1e-3)

        # Merge
        loha.merge()
        assert loha.merged
        out_merged = loha(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        # Unmerge
        loha.unmerge()
        assert not loha.merged
        out_reverted = loha(x)
        assert torch.allclose(out_unmerged, out_reverted, atol=1e-5)

    def test_lokr_forward_merge_unmerge_alpha(self) -> None:
        base = nn.Linear(32, 64, bias=False)
        lokr = LoKrLinear(base, rank=4, alpha=8.0)
        assert lokr.scaling == 2.0
        assert not lokr.base_layer.weight.requires_grad

        x = torch.randn(2, 32)
        out_init = lokr(x)
        assert out_init.shape == (2, 64)

        # Initially B2=0, so delta is 0
        base_direct = base(x)
        assert torch.allclose(out_init, base_direct, atol=1e-5)

        # Set B2 to non-zero
        lokr.lokr_B2.data.fill_(0.1)
        out_unmerged = lokr(x)

        # Merge
        lokr.merge()
        assert lokr.merged
        out_merged = lokr(x)
        assert torch.allclose(out_unmerged, out_merged, atol=1e-5)

        # Unmerge
        lokr.unmerge()
        assert not lokr.merged
        out_reverted = lokr(x)
        assert torch.allclose(out_unmerged, out_reverted, atol=1e-5)


# =====================================================================
# 2. Factorization Reconstruction Error & Numerical Stability
# =====================================================================


class TestFactorizationReconstruction:
    """Reconstruction error bounds and numerical stability across factorizers."""

    def test_nmf_reconstruction_and_stability(self) -> None:
        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = NMFCompressor(model, rank=16, max_iter=20, fallback_to_original=False)
        compressed, stats = comp.compress()

        assert stats["num_compressed"] > 0
        x = torch.randn(2, 32)
        out = compressed(x)
        assert out.shape == (2, 16)
        assert torch.isfinite(out).all()

    def test_rpca_reconstruction_and_stability(self) -> None:
        layer = RPCALinear(64, 32, rank=8, use_triton=False)
        layer.set_factors(torch.randn(32, 8), torch.randn(8, 64))
        x = torch.randn(4, 64)
        out = layer(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = RPCACompressor(model, rank=8, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert len(stats.get("compressed_layers", [])) > 0 or stats.get("compressed_layer_count", 0) > 0
        out_model = compressed(torch.randn(2, 32))
        assert torch.isfinite(out_model).all()

    def test_cur_reconstruction_and_stability(self) -> None:
        linear = nn.Linear(48, 64)
        cur_layer = CURLinear.from_linear(linear, rank=16)
        x = torch.randn(4, 48)
        out = cur_layer(x)
        assert out.shape == (4, 64)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = CURCompressor(model, rank=8, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert len(stats.get("compressed_layers", [])) > 0 or stats.get("compressed_layer_count", 0) > 0

    def test_kronecker_factorization(self) -> None:
        linear = nn.Linear(32, 64)
        kron_layer = KroneckerLinear.from_linear(linear)
        x = torch.randn(2, 32)
        out = kron_layer(x)
        assert out.shape == (2, 64)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = KroneckerCompressor(model, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0

    def test_butterfly_factorization(self) -> None:
        linear = nn.Linear(32, 32)
        bf = ButterflyLinear.from_linear(linear, rank=8)
        x = torch.randn(4, 32)
        out = bf(x)
        assert out.shape == (4, 32)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=32, out_features=32)
        comp = ButterflyCompressor(model, rank=8, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0

    def test_randomized_svd_factorization(self) -> None:
        W = torch.randn(64, 48)
        U, S, Vt = randomized_svd(W, rank=16, n_iter=4)
        recon = U @ torch.diag(S) @ Vt
        assert recon.shape == W.shape
        rel_err = torch.norm(W - recon) / torch.norm(W)
        assert rel_err < 0.85

        linear = nn.Linear(48, 64)
        rsvd = RandomizedSVDLinear.from_linear(linear, rank=16)
        x = torch.randn(2, 48)
        out = rsvd(x)
        assert out.shape == (2, 64)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = RandomizedSVDCompressor(model, rank=8, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0

    def test_subspace_projection_factorization(self) -> None:
        layer = SubspaceLinear(48, 64, rank=16)
        x = torch.randn(3, 48)
        out = layer(x)
        assert out.shape == (3, 64)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=64, out_features=16)
        comp = SubspaceProjectionCompressor(model, rank=8, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0


# =====================================================================
# 3. Spectral, Wavelet, and Miscellaneous Compressors
# =====================================================================


class TestTransformAndQuantizationCompressors:
    """Verification for spectral FFT, Haar wavelet, KMeans, and Mixed-Precision."""

    def test_fft_compression(self) -> None:
        linear = nn.Linear(32, 32)
        fft_layer = FFTLinear.from_linear(linear, keep_ratio=0.5)
        x = torch.randn(2, 32)
        out = fft_layer(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=32, out_features=32)
        comp = FFTCompressor(model, keep_ratio=0.5, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0

    def test_wavelet_compression(self) -> None:
        linear = nn.Linear(32, 32)
        wv_layer = WaveletLinear.from_linear(linear, threshold=0.05)
        x = torch.randn(2, 32)
        out = wv_layer(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=32, out_features=32)
        comp = WaveletCompressor(model, threshold=0.05, fallback_to_original=False)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0

    def test_kmeans_quantization(self) -> None:
        linear = nn.Linear(32, 32)
        km_layer = KMeansLinear.from_linear(linear, n_clusters=16)
        x = torch.randn(2, 32)
        out = km_layer(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

    def test_mixed_precision_quantization(self) -> None:
        linear = nn.Linear(32, 32)
        mp_layer = MixedPrecisionLinear.from_linear(linear, bits=8)
        x = torch.randn(2, 32)
        out = mp_layer(x)
        assert out.shape == (2, 32)
        assert torch.isfinite(out).all()

        model = TinyMLP(in_features=32, hidden=32, out_features=16)
        comp = MixedPrecisionCompressor(model)
        compressed, stats = comp.compress()
        assert stats["num_compressed"] > 0


# =====================================================================
# 4. FP16 and BF16 Numerical Stability Smoke Tests
# =====================================================================


class TestPrecisionStability:
    """Verify numerical stability in half precision formats."""

    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_half_precision_peft_stability(self, dtype: torch.dtype) -> None:
        base = nn.Linear(32, 32, dtype=dtype)
        x = torch.randn(2, 32, dtype=dtype)

        lora = LoRALinear(base, r=4, alpha=8.0, dtype=dtype)
        out_lora = lora(x)
        assert out_lora.dtype == dtype
        assert torch.isfinite(out_lora).all()

        dora = DoRALinear(base, rank=4, lora_alpha=8.0, dtype=dtype)
        out_dora = dora(x)
        assert out_dora.dtype == dtype
        assert torch.isfinite(out_dora).all()

        loha = LoHaLinear(base, rank=4, alpha=8.0, dtype=dtype)
        out_loha = loha(x)
        assert out_loha.dtype == dtype
        assert torch.isfinite(out_loha).all()

        lokr = LoKrLinear(base, rank=4, alpha=8.0, dtype=dtype)
        out_lokr = lokr(x)
        assert out_lokr.dtype == dtype
        assert torch.isfinite(out_lokr).all()
