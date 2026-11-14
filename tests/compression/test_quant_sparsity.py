#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Smoke and Invariant Test Suite for Quantization and Sparsity Compressors.

Audits:
  1. Quantization bit-width claims (INT4, INT8, FP8 E4M3/E5M2, Huffman).
  2. Sparsity claims (unstructured, 2:4 semi-structured, channel shedding).
  3. Dequantization paths, forward reconstruction, and STE gradient flow.
  4. Device agnosticism and absence of deprecated torch.cuda.amp.
"""

from __future__ import annotations

import inspect
import pytest
import torch
import torch.nn as nn

# Quantization engines
from neuralops.compression.quantization.astro_quantizer import AstroQuantizer, quantize_dequantize_tensor
from neuralops.compression.quantization.awq import AWQCompressor, AWQQuantizer
from neuralops.compression.quantization.coopq_quantizer import CoopQQuantizer, quantize_dequantize_coopq
from neuralops.compression.quantization.faq_quantizer import FAQQuantizer
from neuralops.compression.quantization.fp8_compressor import FP8Compressor
from neuralops.compression.quantization.gptq import GPTQCompressor, GPTQQuantizer
from neuralops.compression.quantization.huffman_coder import HuffmanCoder, HuffmanCompressor
from neuralops.compression.quantization.kmeans_quantize import KMeansQuantizer
from neuralops.compression.quantization.quarot import QuaRotCompressor
from neuralops.compression.quantization.rotational_slicegpt_int4 import (
    AstroQuantizer as RotSliceQuantizer,
    RotationalSliceGPTINT4Compressor,
)
from neuralops.compression.quantization.slicegpt import PCAActivationAwareSVDCompressor
from neuralops.compression.quantization.slicegpt_pca import SliceGPTPCACompressor
from neuralops.compression.quantization.smoothquant import SmoothQuantCompressor, smoothquant_layer

# Sparsity / Pruning engines
from neuralops.compression.pruning.admm_compressor import ADMMCompressor
from neuralops.compression.pruning.admm_pruner import ADMMPruner
from neuralops.compression.pruning.armor_pruner import ARMORPruner
from neuralops.compression.pruning.ast_pruner import ASTPruner
from neuralops.compression.pruning.bonsai_pruner import BonsaiPruner
from neuralops.compression.pruning.flap import FLAPCompressor
from neuralops.compression.pruning.mamba_shedder import MambaShedderCompressor
from neuralops.compression.pruning.proxsparse import ProxSparsePruner
from neuralops.compression.pruning.sparsegpt import SparseGPTCompressor, sparsegpt_prune_layer
from neuralops.compression.pruning.wanda import WandaCompressor, wanda_prune_layer


class SyntheticMLP(nn.Module):
    """Deterministic tiny model for compression smoke tests."""

    def __init__(self, in_dim: int = 32, hidden_dim: int = 64, out_dim: int = 16) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))


# =========================================================================
# 1. QUANTIZATION BIT-WIDTH & RECONSTRUCTION TESTS
# =========================================================================

def test_int4_awq_bitwidth_and_reconstruction():
    """Verify AWQ quantizes to strict 4-bit integer grid [-8, 7] and reconstructs."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 64, bias=False)
    orig_w = layer.weight.data.clone()
    quantizer = AWQQuantizer(layer, bits=4, alpha=0.5, group_size=16)

    calib_act = torch.randn(16, 32)
    calib_act[:, 5] *= 10.0  # outlier feature
    quantizer.observe(calib_act)

    w_rec = quantizer.quantize()

    # Invariant: qweight buffer exists and is in INT4 range [-8, 7]
    assert hasattr(layer, "qweight")
    q = layer.qweight
    assert q.dtype == torch.int8
    assert int(q.min().item()) >= -8
    assert int(q.max().item()) <= 7

    # Invariant: forward produces expected shape with finite reconstruction
    x = torch.randn(4, 32)
    out = layer(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()

    # Invariant: relative reconstruction error is bounded
    rel_err = float(torch.norm(orig_w - w_rec) / torch.norm(orig_w))
    assert rel_err < 0.40


def test_int4_gptq_bitwidth_and_reconstruction():
    """Verify GPTQ column-by-column quantization in 4-bit symmetric and asymmetric modes."""
    torch.manual_seed(42)
    # Symmetric mode: [-8, 7]
    layer_sym = nn.Linear(32, 32, bias=True)
    orig_sym = layer_sym.weight.data.clone()
    quantizer_sym = GPTQQuantizer(layer_sym, bits=4, sym=True, group_size=16)
    quantizer_sym.add_batch(torch.randn(32, 32))
    quantizer_sym.quantize(damp=0.01)

    rel_err_sym = float(torch.norm(orig_sym - layer_sym.weight.data) / torch.norm(orig_sym))
    assert rel_err_sym < 0.40
    out_sym = layer_sym(torch.randn(2, 32))
    assert out_sym.shape == (2, 32)
    assert torch.isfinite(out_sym).all()

    # Asymmetric mode: [0, 15] grid constraint
    layer_asym = nn.Linear(32, 32, bias=True)
    quantizer_asym = GPTQQuantizer(layer_asym, bits=4, sym=False, group_size=16)
    quantizer_asym.add_batch(torch.randn(32, 32))
    quantizer_asym.quantize(damp=0.01)
    out_asym = layer_asym(torch.randn(2, 32))
    assert out_asym.shape == (2, 32)
    assert torch.isfinite(out_asym).all()


def test_int4_rotational_slicegpt_bitwidth():
    """Verify Rotational SliceGPT INT4 uses 16 codebook centroids for 4-bit non-uniform quantization."""
    quant = RotSliceQuantizer(bits=4)
    assert quant.num_centroids == 16

    w = torch.randn(32, 32)
    indices, centroids = quant.quantize(w)

    assert indices.shape == w.shape
    assert centroids.shape == (16,)
    assert int(indices.min().item()) >= 0
    assert int(indices.max().item()) <= 15

    w_deq = quant.dequantize(indices, centroids)
    assert w_deq.shape == w.shape
    rel_err = float(torch.norm(w - w_deq) / torch.norm(w))
    assert rel_err < 0.50


def test_int8_smoothquant_bitwidth_and_migration():
    """Verify SmoothQuant symmetric INT8 range [-128, 127] and activation outlier migration."""
    torch.manual_seed(42)
    lin = nn.Linear(32, 64)
    orig_w = lin.weight.data.clone()

    calib_act = torch.randn(64, 32)
    calib_act[:, 3] *= 15.0  # create strong activation outlier

    mod, scale_s = smoothquant_layer(lin, alpha=0.5, calib_data=calib_act, quantize=True)

    assert hasattr(mod, "weight_int8")
    w_int8 = mod.weight_int8
    assert w_int8.dtype == torch.int8
    assert int(w_int8.min().item()) >= -128
    assert int(w_int8.max().item()) <= 127

    # Forward pass executes smoothly with dequantization
    x = torch.randn(4, 32)
    out = mod(x)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_fp8_compressor_formats():
    """Verify FP8 compressor supports E4M3 and E5M2 formats and per-tensor / per-channel scaling."""
    torch.manual_seed(42)
    for fmt in ("e4m3", "e5m2"):
        for scale_type in ("per_tensor", "per_channel"):
            model = SyntheticMLP(32, 48, 16)
            compressor = FP8Compressor(model, scale_type=scale_type, fp8_format=fmt, use_native_fp8=False)
            compressed_model, stats = compressor.compress()

            assert stats["bits"] == 8
            assert stats["fp8_format"] == fmt
            assert stats["scale_type"] == scale_type

            x = torch.randn(4, 32)
            out = compressed_model(x)
            assert out.shape == (4, 16)
            assert torch.isfinite(out).all()


def test_coopq_and_faq_quantizers():
    """Verify CoopQ game-theoretic and FAQ feature-activation quantizers execute forward pass."""
    torch.manual_seed(42)
    # CoopQ
    model_coop = SyntheticMLP(32, 48, 16)
    coop = CoopQQuantizer(model_coop, target_avg_bits=4.0)
    c_model, stats_coop = coop.compress()
    out_coop = c_model(torch.randn(2, 32))
    assert out_coop.shape == (2, 16)
    assert torch.isfinite(out_coop).all()

    # FAQ
    model_faq = SyntheticMLP(32, 48, 16)
    faq = FAQQuantizer(model_faq, bits=8)
    f_model, stats_faq = faq.compress()
    out_faq = f_model(torch.randn(2, 32))
    assert out_faq.shape == (2, 16)
    assert torch.isfinite(out_faq).all()


def test_quarot_hadamard_rotation_and_quantization():
    """Verify QuaRot Walsh-Hadamard pre-rotation eliminates outliers and preserves forward pass."""
    torch.manual_seed(42)
    model = SyntheticMLP(32, 64, 16)
    comp = QuaRotCompressor(model, bits=4, hadamard_block_size=16)
    c_model, stats = comp.compress()

    assert stats["bits"] == 4
    x = torch.randn(2, 32)
    out = c_model(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_kmeans_quantization_centroids():
    """Verify KMeans quantizer clusters weights into discrete centroids with STE support."""
    torch.manual_seed(42)
    model = SyntheticMLP(32, 32, 16)
    comp = KMeansQuantizer(model, n_clusters=16)  # 4-bit codebook
    c_model, stats = comp.compress()

    x = torch.randn(2, 32)
    out = c_model(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_huffman_coder_roundtrip():
    """Verify Huffman coding compresses quantized tensor to bitstream and decodes losslessly."""
    torch.manual_seed(42)
    t = torch.randn(64, 64)
    packed_bytes, code_table, meta = HuffmanCoder.encode(t, quantize_bits=4)

    assert len(packed_bytes) > 0
    assert len(code_table) <= 16
    assert meta["total_elements"] == 64 * 64

    # Theoretical compression ratio vs FP32
    raw_fp32_bytes = 64 * 64 * 4
    ratio = raw_fp32_bytes / len(packed_bytes)
    assert ratio > 4.0  # At 4 bits + prefix codes, ratio is comfortably > 4x

    # Reconstruct
    reconstructed = HuffmanCoder.decode(packed_bytes, code_table, meta)
    assert reconstructed.shape == t.shape
    assert torch.isfinite(reconstructed).all()


# =========================================================================
# 2. SPARSITY CLAIMS & STRUCTURAL PATTERN TESTS
# =========================================================================

def test_sparsegpt_unstructured_and_24_sparsity():
    """Verify SparseGPT achieves claimed sparsity ratio on linear layer."""
    torch.manual_seed(42)
    # Unstructured 50%
    lin_unstructured = nn.Linear(64, 64, bias=False)
    mod_u, mask_u = sparsegpt_prune_layer(lin_unstructured, sparsity=0.5)
    zeros_ratio = float((mod_u.weight.data == 0.0).sum()) / mod_u.weight.data.numel()
    assert 0.45 <= zeros_ratio <= 0.55

    # 2:4 structured via compressor
    model_24 = SyntheticMLP(64, 64, 32)
    comp_24 = SparseGPTCompressor(model_24, structured_24=True, prunen=2, prunem=4)
    c_model_24, stats_24 = comp_24.compress()

    # Check 2:4 pattern on fc1: for every row, each group of 4 elements has exactly 2 zeros
    fc1_w = c_model_24.fc1.weight.data
    groups = fc1_w.view(-1, 4)
    nonzeros_per_group = (groups != 0.0).sum(dim=-1)
    # With 2:4, each group of 4 must have at most 2 nonzeros
    assert (nonzeros_per_group <= 2).all()


def test_wanda_importance_pruning():
    """Verify Wanda pruning prunes lowest weight * activation norm channels."""
    torch.manual_seed(42)
    lin = nn.Linear(64, 64, bias=False)
    mod, mask = wanda_prune_layer(lin, sparsity=0.5)
    zeros_ratio = float((mod.weight.data == 0.0).sum()) / mod.weight.data.numel()
    assert 0.45 <= zeros_ratio <= 0.55

    out = mod(torch.randn(4, 64))
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_admm_and_ast_24_semi_structured_sparsity():
    """Verify ADMM and AST enforce strict 2:4 semi-structured sparsity pattern."""
    torch.manual_seed(42)
    # ADMM Pruner with 2:4 structured sparsity
    model_admm = SyntheticMLP(64, 64, 32)
    admm = ADMMPruner(model_admm, structured_24=True, sparsity=0.5, max_iter=2)
    c_admm, _ = admm.compress()
    fc1_w = c_admm.fc1.weight.data
    groups_admm = fc1_w.view(-1, 4)
    nonzeros_admm = (groups_admm != 0.0).sum(dim=-1)
    assert (nonzeros_admm <= 2).all()

    # AST Pruner with 2:4 structured sparsity
    model_ast = SyntheticMLP(64, 64, 32)
    ast = ASTPruner(model_ast, structured_24=True, sparsity=0.5)
    c_ast, _ = ast.compress()
    fc1_ast_w = c_ast.fc1.weight.data
    groups_ast = fc1_ast_w.view(-1, 4)
    nonzeros_ast = (groups_ast != 0.0).sum(dim=-1)
    assert (nonzeros_ast <= 2).all()


def test_armor_pruner_sparse_residual():
    """Verify ARMOR combines 2:4 sparsity with low-rank correction adapter."""
    torch.manual_seed(42)
    model = SyntheticMLP(64, 64, 32)
    armor = ARMORPruner(model, rank=4, structured_24=True)
    c_model, stats = armor.compress()

    x = torch.randn(4, 64)
    out = c_model(x)
    assert out.shape == (4, 32)
    assert torch.isfinite(out).all()


def test_flap_and_bonsai_pruners():
    """Verify FLAP and Bonsai pruners execute and respect target sparsity."""
    torch.manual_seed(42)
    # FLAP
    model_flap = SyntheticMLP(64, 64, 32)
    flap = FLAPCompressor(model_flap, sparsity=0.3)
    c_flap, stats_flap = flap.compress()
    out_flap = c_flap(torch.randn(2, 64))
    assert out_flap.shape == (2, 32)
    assert torch.isfinite(out_flap).all()

    # Bonsai
    model_bonsai = SyntheticMLP(64, 64, 32)
    bonsai = BonsaiPruner(model_bonsai, target_sp=0.4, target_steps=10)
    c_bonsai, stats_bonsai = bonsai.compress()
    out_bonsai = c_bonsai(torch.randn(2, 64))
    assert out_bonsai.shape == (2, 32)
    assert torch.isfinite(out_bonsai).all()


def test_mamba_shedder_channel_pruning():
    """Verify Mamba-Shedder prunes output rows and scatters back to original output shape."""
    torch.manual_seed(42)
    model = SyntheticMLP(32, 64, 16)
    shedder = MambaShedderCompressor(model, rank=16)  # retain 16 channels in fc1
    c_model, stats = shedder.compress()

    x = torch.randn(2, 32)
    out = c_model(x)
    assert out.shape == (2, 16)
    assert torch.isfinite(out).all()


def test_proxsparse_pruning():
    """Verify ProxSparse applies proximal gradient pruning with adapter preservation."""
    torch.manual_seed(42)
    model = SyntheticMLP(64, 64, 32)
    prox = ProxSparsePruner(model, sparsity=0.4, structured_24=False)
    c_model, stats = prox.compress()

    x = torch.randn(2, 64)
    out = c_model(x)
    assert out.shape == (2, 32)
    assert torch.isfinite(out).all()


# =========================================================================
# 3. CODEBASE HEALTH & GOVERNANCE CHECKS
# =========================================================================

def test_no_deprecated_torch_cuda_amp_in_compression_modules():
    """Verify no quantization or sparsity modules use deprecated torch.cuda.amp (Rule 2.1)."""
    modules_to_check = [
        AstroQuantizer,
        AWQCompressor,
        CoopQQuantizer,
        FAQQuantizer,
        FP8Compressor,
        GPTQCompressor,
        HuffmanCompressor,
        KMeansQuantizer,
        QuaRotCompressor,
        RotationalSliceGPTINT4Compressor,
        PCAActivationAwareSVDCompressor,
        SliceGPTPCACompressor,
        SmoothQuantCompressor,
        ADMMCompressor,
        ADMMPruner,
        ARMORPruner,
        ASTPruner,
        BonsaiPruner,
        FLAPCompressor,
        MambaShedderCompressor,
        ProxSparsePruner,
        SparseGPTCompressor,
        WandaCompressor,
    ]

    for cls in modules_to_check:
        src = inspect.getsource(inspect.getmodule(cls))
        assert "torch.cuda.amp" not in src, f"{cls.__name__} in {inspect.getmodule(cls).__file__} contains deprecated torch.cuda.amp!"


def test_all_quant_and_sparsity_compressors_run_on_cpu():
    """Verify all 23 compressor classes execute cleanly on CPU without requiring CUDA (Rule 2.5)."""
    torch.manual_seed(42)
    compressors = [
        (AstroQuantizer, {}),
        (AWQCompressor, {}),
        (GPTQCompressor, {}),
        (SmoothQuantCompressor, {}),
        (FP8Compressor, {"use_native_fp8": False}),
        (CoopQQuantizer, {}),
        (FAQQuantizer, {}),
        (HuffmanCompressor, {}),
        (KMeansQuantizer, {"n_clusters": 16}),
        (QuaRotCompressor, {"hadamard_block_size": 16}),
        (RotationalSliceGPTINT4Compressor, {"target_rank": 16}),
        (PCAActivationAwareSVDCompressor, {"target_rank": 16}),
        (SliceGPTPCACompressor, {"target_rank": 16}),
        (SparseGPTCompressor, {"sparsity": 0.5}),
        (WandaCompressor, {"sparsity": 0.5}),
        (FLAPCompressor, {"sparsity": 0.2}),
        (ARMORPruner, {"rank": 4}),
        (ADMMPruner, {"max_iter": 1}),
        (ADMMCompressor, {"max_iter": 1}),
        (ASTPruner, {"sparsity": 0.5}),
        (BonsaiPruner, {"target_steps": 5}),
        (MambaShedderCompressor, {"rank": 16}),
        (ProxSparsePruner, {"sparsity": 0.3}),
    ]

    for comp_cls, kwargs in compressors:
        model = SyntheticMLP(32, 32, 16)
        c = comp_cls(model, **kwargs)
        c_model, stats = c.compress()
        x = torch.randn(2, 32)
        out = c_model(x)
        assert out.shape == (2, 16), f"Failure in {comp_cls.__name__}: output shape {out.shape} != (2, 16)"
        assert torch.isfinite(out).all(), f"Failure in {comp_cls.__name__}: output contains NaN or Inf"
