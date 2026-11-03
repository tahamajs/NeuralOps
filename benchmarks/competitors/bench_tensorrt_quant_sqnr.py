#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TensorRT-LLM Microscaling & SQNR Quantization Distortion Benchmark Harness.

Evaluates Signal-to-Quantization-Noise Ratio (SQNR) in decibels (dB), relative Frobenius norm distortion,
and maximum absolute error across FP8 (e4m3fn / e5m2), INT8, and INT4 weight quantization.
Originating standard: NVIDIA / PyTorch & TensorRT Team.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from neuralops.benchmarks.registry import register_benchmark

logger = logging.getLogger(__name__)


def benchmark_tensorrt_sqnr(
    original_tensor: torch.Tensor,
    quantized_tensor: torch.Tensor,
    min_sqnr_threshold_db: float = 32.0,
    max_frobenius_threshold: float = 0.03,
) -> dict[str, Any]:
    """Calculate Signal-to-Quantization-Noise Ratio (SQNR) and Frobenius distortion.

    Args:
        original_tensor: Reference full-precision tensor.
        quantized_tensor: Dequantized or compressed reconstruction tensor.
        min_sqnr_threshold_db: Gate for minimum acceptable SQNR in dB.
        max_frobenius_threshold: Gate for maximum allowable relative Frobenius distortion.

    Returns:
        Dictionary with sqnr_db, rel_frobenius, max_abs_diff, and passed boolean gate.
    """
    orig = original_tensor.float()
    quant = quantized_tensor.float()
    diff = orig - quant

    signal_pow = torch.sum(orig ** 2)
    noise_pow = torch.sum(diff ** 2) + 1e-12

    sqnr_db = 10.0 * torch.log10(signal_pow / noise_pow).item()
    rel_frobenius = (torch.norm(diff) / (torch.norm(orig) + 1e-12)).item()
    max_abs_diff = torch.max(torch.abs(diff)).item()

    passed = (sqnr_db >= min_sqnr_threshold_db) and (rel_frobenius <= max_frobenius_threshold)

    return {
        "sqnr_db": sqnr_db,
        "rel_frobenius": rel_frobenius,
        "max_abs_diff": max_abs_diff,
        "passed": passed,
    }


def quantize_simulate(
    tensor: torch.Tensor,
    format_type: str = "fp8_e4m3",
    block_size: int = 32,
) -> torch.Tensor:
    """Simulate low-precision quantization (FP8, INT8, INT4) with scale factor.

    Supports:
        - 'fp8_e4m3': FP8 with 1 sign bit, 4 exponent bits, 3 mantissa bits (max ~448).
        - 'fp8_e5m2': FP8 with 1 sign bit, 5 exponent bits, 2 mantissa bits (max ~57344).
        - 'int8': Symmetric per-tensor or per-channel 8-bit signed integer (-128 to 127).
        - 'int4': Symmetric 4-bit signed integer (-8 to 7).
        - 'int4_mx': OCP MX microscaling standard INT4 with 32-element scaling blocks.
    """
    t = tensor.float()
    if format_type == "fp8_e4m3":
        # Simulates 8-bit float with 1 sign, 4 exponent, 3 mantissa (max ~448)
        max_val = 448.0
        scale = max_val / (t.abs().max().clamp(min=1e-8))
        scaled = (t * scale).clamp(-max_val, max_val)
        # Round to 3-bit mantissa resolution
        step = 2.0 ** (-3)
        quant = torch.round(scaled / step) * step
        return quant / scale
    elif format_type == "fp8_e5m2":
        # Simulates 8-bit float with 1 sign, 5 exponent, 2 mantissa (max ~57344)
        max_val = 57344.0
        scale = max_val / (t.abs().max().clamp(min=1e-8))
        scaled = (t * scale).clamp(-max_val, max_val)
        step = 2.0 ** (-2)
        quant = torch.round(scaled / step) * step
        return quant / scale
    elif format_type == "int8":
        scale = 127.0 / (t.abs().max().clamp(min=1e-8))
        quant = torch.clamp(torch.round(t * scale), -128, 127)
        return quant / scale
    elif format_type == "int4":
        scale = 7.0 / (t.abs().max().clamp(min=1e-8))
        quant = torch.clamp(torch.round(t * scale), -8, 7)
        return quant / scale
    elif format_type in ("int4_mx", "int4_microscaling"):
        # MX microscaling standard: block-wise scaling with block_size elements (default 32)
        orig_shape = t.shape
        numel = t.numel()
        pad_len = (block_size - (numel % block_size)) % block_size
        if pad_len > 0:
            flat = F.pad(t.reshape(-1), (0, pad_len))
        else:
            flat = t.reshape(-1)

        blocks = flat.view(-1, block_size)
        block_max = blocks.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
        scale = 7.0 / block_max
        quant_blocks = torch.clamp(torch.round(blocks * scale), -8, 7) / scale
        dequant_flat = quant_blocks.view(-1)[:numel]
        return dequant_flat.view(orig_shape)
    else:
        raise ValueError(f"Unsupported quantization format: {format_type}")


@register_benchmark(name="tensorrt_quant_sqnr", mode="sweep-compressors", hardware="any")
def run_tensorrt_quant_sqnr_suite(
    layer_shapes: list[tuple[int, int]] | None = None,
    formats: list[str] | None = None,
    device: str | torch.device = "cpu",
    seed: int = 42,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run TensorRT quantization SQNR distortion benchmark suite.

    Args:
        layer_shapes: List of (in_features, out_features) projection weight shapes.
        formats: List of formats to benchmark (e.g. ['fp8_e4m3', 'int8', 'int4']).
        device: Device to allocate test tensors.
        seed: Random seed for deterministic tensor generation.
        **kwargs: Additional parameters.

    Returns:
        Aggregated results dictionary with per-layer SQNR and distortion gate pass/fail.
    """
    dev = torch.device(device)
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested for run_tensorrt_quant_sqnr_suite, but CUDA is not available."
        )

    torch.manual_seed(seed)
    if layer_shapes is None:
        layer_shapes = [(4096, 4096), (4096, 11008), (11008, 4096)]
    if formats is None:
        formats = ["fp8_e4m3", "int8", "int4"]

    results_by_layer: list[dict[str, Any]] = []
    all_passed = True

    for in_dim, out_dim in layer_shapes:
        w_orig = torch.randn(out_dim, in_dim, device=dev, dtype=torch.float32)
        layer_res: dict[str, Any] = {
            "shape": f"{out_dim}x{in_dim}",
            "formats": {},
        }
        for fmt in formats:
            w_quant = quantize_simulate(w_orig, format_type=fmt)
            distortion = benchmark_tensorrt_sqnr(w_orig, w_quant)
            layer_res["formats"][fmt] = distortion
            if not distortion["passed"] and fmt in ("fp8_e4m3", "int8"):
                all_passed = False

        results_by_layer.append(layer_res)

    return {
        "benchmark": "tensorrt_quant_sqnr",
        "evaluated_layers_count": len(layer_shapes),
        "formats_tested": formats,
        "layers": results_by_layer,
        "all_standard_gates_passed": all_passed,
    }


def run_benchmark(**kwargs: Any) -> dict[str, Any]:
    """Standard entrypoint wrapper for benchmark runner."""
    return run_tensorrt_quant_sqnr_suite(**kwargs)


# Alias for explicit phase naming compliance
run_tensorrt_quant_benchmark = run_tensorrt_quant_sqnr_suite
CoreModule = run_tensorrt_quant_sqnr_suite

