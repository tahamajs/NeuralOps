#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Comprehensive verification tests for concrete quantization and structured pruning implementations.

Verifies:
1. SmoothQuant: INT8 symmetric quantization, scale parameters, and true quantized module operation.
2. GPTQ: Robust Tikhonov regularization, ill-conditioned Hessian inversion, and honest telemetry.
3. AWQ: Retention of integer grid representation, explicit scaling factors, and honest telemetry.
4. ADMM: Explicit binary masks (2:4 and unstructured) and true active non-zero weight count telemetry.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.compression.pruning.admm_compressor import ADMMCompressor
from neuralops.compression.quantization.awq import AWQCompressor, AWQQuantizer
from neuralops.compression.quantization.gptq import GPTQCompressor, GPTQQuantizer
from neuralops.compression.quantization.smoothquant import (
    SmoothQuantCompressor,
    SmoothQuantLinear,
    smoothquant_layer,
)


def test_smoothquant_int8_quantization_and_scales():
    """Verify SmoothQuant layer produces true INT8 weights and registers scaling buffers."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 16, bias=True)
    x = torch.randn(8, 32)
    x[:, 3] *= 8.0  # activation outlier

    mod, s = smoothquant_layer(layer, alpha=0.5, calib_data=x, quantize=True)

    # 1. Buffers attached
    assert hasattr(mod, "weight_int8")
    assert hasattr(mod, "weight_scale")
    assert hasattr(mod, "s_inv")
    assert hasattr(mod, "act_scale")

    # 2. Strict INT8 representation
    assert mod.weight_int8.dtype == torch.int8
    assert mod.weight_int8.min().item() >= -128
    assert mod.weight_int8.max().item() <= 127

    # 3. Output matches scaled linear GEMM
    out = mod(x)
    assert out.shape == (8, 16)
    scaled_x = x * mod.s_inv.unsqueeze(0)
    manual_out = F.linear(scaled_x, mod.weight, mod.bias)
    assert torch.allclose(out, manual_out, atol=1e-4)


def test_smoothquant_linear_module_execution():
    """Verify SmoothQuantLinear standalone module operates with true quantized buffers."""
    torch.manual_seed(42)
    s = torch.ones(32) * 1.2
    mod = SmoothQuantLinear(in_features=32, out_features=16, s=s, bias=True)

    assert hasattr(mod, "weight_int8")
    assert mod.weight_int8.dtype == torch.int8
    assert hasattr(mod, "weight_scale")
    assert hasattr(mod, "s_inv")

    x = torch.randn(4, 32)
    out = mod(x)
    assert out.shape == (4, 16)


def test_smoothquant_compressor_telemetry():
    """Verify SmoothQuantCompressor records honest telemetry and exports buffers."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(4, 32) for _ in range(2)]

    compressor = SmoothQuantCompressor(model, alpha=0.5, calib_data=calib)
    comp_model, stats = compressor.compress()

    assert "layers" in stats
    assert len(stats["layers"]) == 2
    assert "layer_errors" in stats
    assert "avg_reconstruction_error" in stats
    assert stats["compression_ratio"] == 2.0  # 16 / 8

    tensors = compressor.compress_to_tensors()
    assert any("weight_int8" in k for k in tensors.keys())
    assert any("weight_scale" in k for k in tensors.keys())


def test_gptq_tikhonov_regularization_and_ill_conditioned():
    """Verify GPTQ handles ill-conditioned/singular Hessians using robust Tikhonov regularization."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 16, bias=False)
    quantizer = GPTQQuantizer(layer, bits=4, group_size=16)

    # Singular activation batch (rank 1 matrix: all rows identical)
    rank1_x = torch.ones(8, 32)
    quantizer.add_batch(rank1_x)

    # Invert and quantize: must not crash with singular matrix error
    w_q = quantizer.quantize(damp=0.01)

    assert w_q.shape == (16, 32)
    assert not torch.isnan(w_q).any()
    assert not torch.isinf(w_q).any()
    assert quantizer.error >= 0.0
    assert quantizer.damp_val > 0.0


def test_gptq_compressor_honest_telemetry():
    """Verify GPTQCompressor records rich and honest reconstruction error telemetry."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(8, 32) for _ in range(2)]

    for mod in model.children():
        q = GPTQQuantizer(mod, bits=4)
        for b in calib:
            q.add_batch(b)

    compressor = GPTQCompressor(model, bits=4, damp=0.01)
    comp_model, stats = compressor.compress()

    assert "layers" in stats
    assert "layer_errors" in stats
    assert "avg_reconstruction_error" in stats
    assert "total_parameters" in stats
    assert stats["compression_ratio"] == 4.0  # 16 / 4
    assert stats["avg_reconstruction_error"] >= 0.0


def test_awq_integer_grid_and_explicit_scales():
    """Verify AWQ quantizes strictly to integer grid without un-quantized division."""
    torch.manual_seed(42)
    layer = nn.Linear(32, 16, bias=False)
    awq = AWQQuantizer(layer, bits=4, alpha=0.5, group_size=16)

    x = torch.randn(8, 32)
    awq.observe(x)
    w_deq = awq.quantize()

    assert hasattr(layer, "qweight")
    assert hasattr(layer, "scales")
    assert layer.qweight.dtype == torch.int8

    # Integer grid values must be within [-8, 7] for INT4
    assert layer.qweight.min().item() >= -8
    assert layer.qweight.max().item() <= 7

    # Reconstruction check: W_dequant must equal qweight * scales
    reconstructed = layer.qweight.float() * layer.scales.float()
    assert torch.allclose(w_deq, reconstructed, atol=1e-5)
    assert torch.allclose(layer.weight.data, reconstructed, atol=1e-5)


def test_awq_compressor_telemetry():
    """Verify AWQCompressor captures layer reconstruction telemetry and tensor export."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(32, 32), nn.Linear(32, 16))
    calib = [torch.randn(8, 32) for _ in range(2)]

    compressor = AWQCompressor(model, bits=4, calib_data=calib)
    comp_model, stats = compressor.compress()

    assert "layers" in stats
    assert "layer_errors" in stats
    assert "avg_reconstruction_error" in stats
    assert stats["compression_ratio"] == 4.0

    tensors = compressor.compress_to_tensors()
    assert any("qweight" in k for k in tensors.keys())
    assert any("scales" in k for k in tensors.keys())


def test_admm_explicit_binary_masks_and_telemetry():
    """Verify ADMM generates explicit binary masks and records true active count telemetry."""
    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))

    compressor = ADMMCompressor(
        model,
        sparsity=0.5,
        quant_bits=4,
        structured_24=True,
        max_error=0.8,
        fallback_to_original=False,
    )
    comp_model, stats = compressor.compress()

    # 1. Telemetry keys present
    assert "active_weight_counts" in stats
    assert "total_weight_counts" in stats
    assert "total_active_weights" in stats
    assert "true_compression_ratio" in stats
    assert stats["true_compression_ratio"] >= 1.8

    # 2. Explicit binary masks attached
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert hasattr(module, "pruning_mask")
            assert module.pruning_mask.dtype == torch.bool
            assert module.pruning_mask.shape == module.weight.shape

            active = int(module.pruning_mask.sum().item())
            assert stats["active_weight_counts"][name] == active
            assert (module.weight.data != 0).sum().item() == active

    # 3. Tensors export
    tensors = compressor.compress_to_tensors()
    assert any("pruning_mask" in k for k in tensors.keys())


def test_sparsegpt_explicit_binary_masks_and_telemetry():
    """Verify SparseGPT attaches explicit binary masks and records true active count telemetry."""
    from neuralops.compression.pruning.sparsegpt import SparseGPTCompressor

    torch.manual_seed(42)
    model = nn.Sequential(nn.Linear(64, 64), nn.Linear(64, 32))
    calib = [torch.randn(8, 64) for _ in range(2)]

    compressor = SparseGPTCompressor(
        model,
        sparsity=0.5,
        structured_24=False,
        max_error=0.9,
        fallback_to_original=False,
    )
    # mock activation collection
    compressor._collect_layer_activations = lambda: {
        "0": torch.cat(calib, dim=0),
        "1": torch.randn(16, 64),
    }

    comp_model, stats = compressor.compress()

    assert "active_weight_counts" in stats
    assert "total_weight_counts" in stats
    assert "true_compression_ratio" in stats
    assert stats["true_compression_ratio"] >= 1.5

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            assert hasattr(module, "pruning_mask")
            assert module.pruning_mask.dtype == torch.bool
            assert module.pruning_mask.shape == module.weight.shape
            active = int(module.pruning_mask.sum().item())
            assert stats["active_weight_counts"][name] == active
            assert (module.weight.data != 0).sum().item() == active

    tensors = compressor.compress_to_tensors()
    assert any("pruning_mask" in k for k in tensors.keys())


def test_wanda_binary_mask_attachment():
    """Verify Wanda pruning registers explicit binary masks on modified modules."""
    from neuralops.compression.pruning.wanda import wanda_prune_layer

    torch.manual_seed(42)
    layer = nn.Linear(32, 16)
    calib = torch.randn(8, 32)
    mod, mask = wanda_prune_layer(layer, sparsity=0.5, calib_data=calib)

    assert hasattr(mod, "pruning_mask")
    assert hasattr(mod, "mask")
    assert mod.pruning_mask.dtype == torch.bool
    assert torch.equal(mod.pruning_mask, mask)
    assert int((mod.weight.data != 0).sum().item()) == int(mod.pruning_mask.sum().item())

