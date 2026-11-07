#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Kernels Facade (Lazy PEP 562).
"""

from __future__ import annotations

import importlib
from typing import Any

from neuralops.ops.cuda._map import CUDA_KERNEL_CATALOG, _KERNEL_SYMBOL_MAP, _SPECIAL_MAP

_SUBPACKAGES = ["cuda", "triton", "extended_kernels"]

__all__ = sorted(
    list(
        set(_SUBPACKAGES)
        | set(CUDA_KERNEL_CATALOG)
        | set(_KERNEL_SYMBOL_MAP.keys())
        | set(_SPECIAL_MAP.keys())
        | {
            "HAS_TRITON",
            "triton_dispatch",
            "extended_kernels",
            "fast_attention",
            "cross_attention",
            "sliding_window_attn",
            "fused_rotary_emb",
            "fused_norm_linear",
            "fused_swiglu",
            "fused_rmsnorm",
            "fused_alibi_attention",
            "fused_sink_token_attention",
            "fused_decay_scan",
            "fused_hadamard_transform",
            "fused_int4_dequant",
            "fused_fp8_e4m3_gemm",
            "fused_spectral_dct",
            "fused_chunked_cumsum",
            "fused_cka_alignment",
            "fused_mohawk_homotopy",
            "fused_nullspace_project",
            "fused_householder_reflect",
            "fused_simplex_project",
            "fused_topk_gating",
            "fused_label_smoothing_ce",
            "fused_kl_divergence",
            "fused_lowrank_residual_add",
            "fused_lowrank_gated_gemm",
        }
    )
)


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes lazily from subpackages, ops.cuda, or ops.triton."""
    if name.startswith("_") and name not in _KERNEL_SYMBOL_MAP:
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

    if name in _SUBPACKAGES:
        mod = importlib.import_module(f".{name}", __name__)
        globals()[name] = mod
        return mod

    if name == "HAS_TRITON":
        from neuralops.ops.triton._base import HAS_TRITON

        return HAS_TRITON

    if name == "triton_dispatch":
        from neuralops.ops.triton._common import triton_dispatch

        return triton_dispatch

    # Check extended_kernels
    try:
        from neuralops.kernels import extended_kernels

        if hasattr(extended_kernels, name):
            val = getattr(extended_kernels, name)
            globals()[name] = val
            return val
    except Exception:
        pass

    # Check CUDA ops
    import neuralops.ops.cuda as _ops_cuda

    if hasattr(_ops_cuda, name):
        val = getattr(_ops_cuda, name)
        globals()[name] = val
        return val

    if name in CUDA_KERNEL_CATALOG:
        mod = importlib.import_module(f"neuralops.ops.cuda.{name}")
        globals()[name] = mod
        return mod

    # Check Triton ops
    try:
        import neuralops.ops.triton as _ops_triton

        if hasattr(_ops_triton, name):
            val = getattr(_ops_triton, name)
            globals()[name] = val
            return val
    except Exception:
        pass

    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
