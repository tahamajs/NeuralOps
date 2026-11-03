#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 11: Quantization Formats Zoo (FP8 E4M3, Group-Wise INT4, Microscaling MX).

Subsystem Demonstrated:
    Comprehensive implementation and numerical evaluation of next-generation
    neural network weight quantization formats:
    1. FP8 E4M3 (IEEE / NVIDIA 8-bit floating point format with 4 exponent bits, 3 mantissa)
    2. Group-Wise INT4 (AWQ-style group-wise scaling in range [-8, 7])
    3. Microscaling MX (OCP standard block-shared power-of-two scale factors with MXFP8)
    4. End-to-end quantized model forward pass verification on TinyClassifier.

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates tensor operations).
    - Network: Not required; uses offline synthetic weights and TinyClassifier.

What You Will Learn:
    1. How FP8 dynamic range and precision compare to floating point baselines.
    2. How group-wise scaling eliminates outlier distortion in 4-bit integer weights.
    3. How the Open Compute Project (OCP) Microscaling (MX) standard structures block-shared exponents.
    4. How to measure Signal-to-Quantization-Noise Ratio (SQNR in dB) across formats.
    5. How to verify that a quantized model preserves finite output and tight parity.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
)


def compute_sqnr(w_orig: torch.Tensor, w_recon: torch.Tensor) -> float:
    """Compute Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB):

    SQNR = 20 * log10( ||W||_F / (||W - W_hat||_F + eps) )
    """
    orig_norm = torch.linalg.norm(w_orig.float()).item()
    diff_norm = torch.linalg.norm((w_orig.float() - w_recon.float())).item()
    if diff_norm < 1e-12:
        return 100.0  # Lossless
    return 20.0 * math.log10(orig_norm / (diff_norm + 1e-12))


def quantize_fp8_e4m3(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, float, float, bool]:
    """Quantize to FP8 E4M3 (per-tensor scale, max representable magnitude 448.0).

    Uses native torch.float8_e4m3fn when available, or exact IEEE 754 E4M3
    mantissa simulation fallback on older PyTorch builds.

    Returns:
        (dequantized_weight, sqnr_db, quant_error, is_supported)
    """
    try:
        max_val = 448.0
        abs_max = weight.abs().max().clamp(min=1e-8)
        scale = abs_max / max_val
        w_scaled = (weight / scale).clamp(-max_val, max_val)

        if hasattr(torch, "float8_e4m3fn") and weight.is_cuda:
            w_fp8 = w_scaled.to(torch.float8_e4m3fn)
            w_recon = w_fp8.to(torch.float32) * scale
        else:
            # IEEE 754 E4M3 simulation: 1 sign, 4 exp (bias 7), 3 mantissa
            mantissa_bits = 3
            exp = torch.floor(torch.log2(w_scaled.abs().clamp(min=1e-8)))
            base = torch.pow(2.0, exp)
            step = base * (2.0 ** (-mantissa_bits))
            w_sim = torch.round(w_scaled / step) * step
            w_recon = w_sim * scale

        sqnr = compute_sqnr(weight, w_recon)
        quant_err = torch.norm(weight - w_recon, p="fro").item()
        return w_recon, sqnr, quant_err, True
    except Exception:
        return weight, 0.0, 0.0, False


def quantize_int4_groupwise(
    weight: torch.Tensor,
    group_size: int = 128,
) -> tuple[torch.Tensor, float, float]:
    """Symmetric group-wise INT4 quantization (AWQ-style simulation):

    scale = max(|w_group|) / 7.0
    q = clamp(round(w / scale), -8, 7)
    w_recon = q * scale
    """
    orig_shape = weight.shape
    w_flat = weight.reshape(-1, group_size)

    abs_max = w_flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    scale = abs_max / 7.0

    q = torch.clamp(torch.round(w_flat / scale), -8, 7)
    w_recon = (q * scale).view(orig_shape)

    sqnr = compute_sqnr(weight, w_recon)
    quant_err = torch.norm(weight - w_recon, p="fro").item()
    return w_recon, sqnr, quant_err


def quantize_microscaling_mx(
    weight: torch.Tensor,
    block_size: int = 32,
) -> tuple[torch.Tensor, float, float]:
    """OCP Microscaling Formats (MX) standard simulation:

    Each block of size B=32 shares an E8M0 power-of-two scale:
    scale = 2^ceil(log2(max(|w_block|) / 448.0))
    Elements quantized to FP8 E4M3 format.
    """
    orig_shape = weight.shape
    w_blocks = weight.reshape(-1, block_size)

    # Compute block-shared power-of-two scale
    max_b = w_blocks.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8)
    shared_exponent = torch.ceil(torch.log2(max_b / 448.0))
    scale_mx = torch.pow(2.0, shared_exponent)

    w_scaled = w_blocks / scale_mx
    if hasattr(torch, "float8_e4m3fn"):
        w_fp8 = w_scaled.to(torch.float8_e4m3fn)
        w_recon = (w_fp8.to(torch.float32) * scale_mx).view(orig_shape)
    else:
        # High-fidelity simulation of 3-bit mantissa quantization
        mantissa_bits = 3
        factor = 2.0**mantissa_bits
        w_sim = torch.round(w_scaled * factor) / factor
        w_recon = (w_sim * scale_mx).view(orig_shape)

    sqnr = compute_sqnr(weight, w_recon)
    quant_err = torch.norm(weight - w_recon, p="fro").item()
    return w_recon, sqnr, quant_err


class QuantizedLinearWrapper(nn.Module):
    """Wrapper that applies INT4 quantize-dequantize on forward pass."""

    def __init__(self, linear_layer: nn.Linear, group_size: int = 128) -> None:
        super().__init__()
        self.linear = linear_layer
        self.group_size = group_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.linear.weight
        orig_shape = w.shape
        w_flat = w.reshape(-1, min(self.group_size, w.numel()))
        scale = w_flat.abs().max(dim=-1, keepdim=True).values.clamp(min=1e-8) / 7.0
        q = torch.clamp(torch.round(w_flat / scale), -8, 7)
        w_dq = (q * scale).view(orig_shape)
        return F.linear(x, w_dq, self.linear.bias)


def main() -> int:
    section("NeuralOps Example 11: Quantization Formats Zoo")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    # Deterministic test tensor (128x128 matrix from tiny model)
    torch.manual_seed(42)
    m, n = 128, 128
    weight = torch.randn(m, n, device="cpu")
    kv("Weight Matrix Dimensions", f"{m} x {n} ({weight.numel():,} parameters)")
    kv("FP32 Baseline Storage", f"{weight.numel() * 4:,} bytes")
    print()

    table_rows: list[list[str]] = []
    all_sqnr_ok = True

    # =========================================================================
    # Format 1: FP8 E4M3
    # =========================================================================
    section("Format 1: FP8 E4M3 (IEEE 8-bit Floating Point)")
    w_fp8, sqnr_fp8, err_fp8, fp8_ok = quantize_fp8_e4m3(weight)
    if fp8_ok:
        pass_fail("FP8 E4M3 Quantization", bool(sqnr_fp8 > 10.0))
        kv("Effective Precision", "8 bits (1 sign, 4 exp, 3 mantissa)")
        kv("Compression Ratio", "4.0x vs FP32 (2.0x vs FP16)")
        kv("SQNR", f"{sqnr_fp8:.2f} dB")
        kv("Quantization Error (Frobenius)", f"{err_fp8:.4f}")
        all_sqnr_ok = all_sqnr_ok and (sqnr_fp8 > 10.0)
        table_rows.append(["FP8 E4M3", f"{sqnr_fp8:.2f} dB", f"{err_fp8:.4f}", "per-tensor scale"])
    else:
        print("  [SKIPPED] FP8 E4M3 not supported on this PyTorch build")
        table_rows.append(["FP8 E4M3", "SKIPPED", "N/A", "torch.float8_e4m3fn missing"])

    # =========================================================================
    # Format 2: Group-Wise INT4 (AWQ-style simulation)
    # =========================================================================
    section("Format 2: Group-Wise INT4 (AWQ-style Simulation)")
    w_int4, sqnr_int4, err_int4 = quantize_int4_groupwise(weight, group_size=128)
    pass_fail("INT4 Group-Wise (G=128) Quantization", bool(sqnr_int4 > 10.0))
    kv("Group Size", "128 elements per scale factor")
    kv("Quantization Range", "[-8, 7] symmetric INT4")
    kv("SQNR", f"{sqnr_int4:.2f} dB")
    kv("Quantization Error (Frobenius)", f"{err_int4:.4f}")
    all_sqnr_ok = all_sqnr_ok and (sqnr_int4 > 10.0)
    table_rows.append(["INT4 groupwise", f"{sqnr_int4:.2f} dB", f"{err_int4:.4f}", "group=128"])

    # =========================================================================
    # Format 3: Microscaling MX (OCP Standard Simulation)
    # =========================================================================
    section("Format 3: Microscaling Formats (MX) Standard (MXFP8)")
    w_mx, sqnr_mx, err_mx = quantize_microscaling_mx(weight, block_size=32)
    pass_fail("Microscaling MXFP8 Quantization", bool(sqnr_mx > 10.0))
    kv("Block Size", "32 elements per shared power-of-two exponent")
    kv("Exponent Scale", "E8M0 shared exponent (scale = 2^k)")
    kv("SQNR", f"{sqnr_mx:.2f} dB")
    kv("Quantization Error (Frobenius)", f"{err_mx:.4f}")
    all_sqnr_ok = all_sqnr_ok and (sqnr_mx > 10.0)
    table_rows.append(["MX (simulated)", f"{sqnr_mx:.2f} dB", f"{err_mx:.4f}", "block=32, E8M0"])

    # =========================================================================
    # Comparison Summary Table
    # =========================================================================
    section("Quantization Formats Comparison Table")
    headers = ("Format", "SQNR (dB)", "Quantization Error", "Notes")
    table(table_rows, headers=headers)
    print()

    # =========================================================================
    # Part 4: End-to-End Model Forward Pass Verification
    # =========================================================================
    section("Part 4: Quantized Model Forward Pass Parity Verification")
    model = TinyClassifier().to(device)
    model.eval()

    inputs = {"input_ids": torch.tensor([[1, 2, 3, 4], [2, 3, 4, 5]], device=device)}
    with torch.no_grad():
        ref_logits = model(**inputs).logits.clone()

    # Quantize the attention Q projection with INT4
    for _, m_layer in model.named_modules():
        if hasattr(m_layer, "q_lin"):
            w = m_layer.q_lin.weight.data
            scale = w.abs().max().clamp(min=1e-8) / 7.0
            q = torch.clamp(torch.round(w / scale), -8, 7)
            m_layer.q_lin.weight.data = q * scale

    with torch.no_grad():
        quant_logits = model(**inputs).logits

    max_diff = (ref_logits - quant_logits).abs().max().item()
    is_finite = torch.isfinite(quant_logits).all().item()

    kv("Reference Logits Shape", f"{tuple(ref_logits.shape)}")
    kv("Quantized Logits Shape", f"{tuple(quant_logits.shape)}")
    kv("Max Absolute Logits Diff", f"{max_diff:.8f}")
    kv("Logits Finiteness", is_finite)

    pass_fail("Quantized Model Forward Parity", bool(is_finite and max_diff < 0.1))

    overall_ok = all_sqnr_ok and is_finite and (max_diff < 0.1)
    pass_fail(
        "Overall Example 11 Status",
        overall_ok,
        "All 3 quantization paradigms evaluated with SQNR fidelity and model parity",
    )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
