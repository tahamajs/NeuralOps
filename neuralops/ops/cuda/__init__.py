#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Native CUDA C++ Kernel Operators (Tier 1)

PEP 562 lazy-loading dispatcher resolving kernel symbols on access without
eagerly importing native CUDA packages at import time.
"""

from __future__ import annotations

import importlib
from typing import Any

from neuralops.ops.cuda._map import (
    CUDA_KERNEL_CATALOG,
    _KERNEL_SYMBOL_MAP,
    _SPECIAL_MAP,
)


def list_cuda_kernels(category: str | None = None) -> list[str]:
    """Return list of all available CUDA kernels in NeuralOps.

    Args:
        category: Filter category:
            - None or 'all': All 114 native CUDA kernel implementations (.cu files).
            - 'modules': 110 unique module package names.
            - 'operators': Primary CUDA operator function names.

    Returns:
        Sorted list of kernel names.
    """
    if category == "modules":
        return sorted(list({
            "act_scale_gemm", "adaptive_span", "aegis_gate_norm", "alibi_attention",
            "bigbird", "bitonic_topk", "block_diag_gemm", "causal_conv1d",
            "channel_shuffle", "cluster_attention", "cosformer", "cross_attention",
            "decay_linear_attention", "differential_attention", "dynamic_conv1d",
            "entmax_bisect", "entropy_gate", "entropy_pruning", "fast_attention",
            "fast_gelu", "favor_plus", "flash_decode_stage2", "flash_decoding",
            "flash_gqa", "fms_scan", "fp4_gemm", "fp8_gemm", "fused_ce",
            "fused_dropout_add", "fused_kd", "fused_norm_linear", "fused_rope",
            "fwht_transform", "galerkin_attention", "gdn_delta_rule", "gdn_scan",
            "gla_scan", "gla_unnormalized_scan", "goldfinch_ahn", "griffin_ssd",
            "hadamard_gemm", "hgrn2_scan", "hgrn2_spatial_scan", "hstu_attention",
            "hydra_attention", "hymba_transx", "jamba", "jamba_info",
            "kronecker_gemm", "latent_kv_attn", "layernorm", "linear_attention",
            "linear_rnn_attention", "linformer_attention", "liquid_attention",
            "local_attention", "logcumsumexp", "longformer", "low_rank_svd",
            "lut_gemm", "mamba2_block", "mamba_block", "mamba_svd_flash",
            "moe_routing", "multi_scale_retention", "nasc_attention", "nexus",
            "nystrom_attention", "orion", "paged_attention", "paged_kv_cache",
            "paged_kv_evict", "paged_recurrent_step", "phantom", "quantized_gemm",
            "recurrent_gla_step", "recurrent_hgrn2_step", "reformer", "rfa_attention",
            "rms_variance_match", "rmsnorm", "rotational_svd", "routing_attention",
            "rpca_sparse_svd", "rwkv6", "samba_m3", "samba_moe", "selective_scan",
            "selective_ssm", "simple_ssm", "sliding_window", "soa_attn",
            "softplus", "sparse_attention", "spectra_v2", "spectraformer_attention",
            "spectral_mamba", "spectral_shrinkage", "speculative_verify",
            "split_head_norm", "state_update", "svd_prefix_scan", "swiglu",
            "ternary_gemm", "tpa_attention", "tt_contraction", "unified_rope_offset",
            "upi_stuffed_mamba", "varlen_attention", "vspec",
        }))
    if category == "operators":
        import neuralops.ops.cuda as nck
        util_names = {
            "load_cuda_module", "get_cuda_arch_flags", "dispatch", "force_kernel_backend",
            "get_dispatch_stats", "reset_dispatch_stats", "is_cuda_available",
            "get_device_capability", "check_cuda_error", "list_cuda_kernels",
        }
        return sorted([
            k for k in nck.__all__
            if k not in util_names
            and not k.endswith("_pytorch_ref")
            and not k.endswith("_pytorch_reference")
            and not k.endswith("_pytorch")
            and not k.endswith("_ref")
        ])
    return list(CUDA_KERNEL_CATALOG)


def __getattr__(name: str) -> Any:
    if name in _KERNEL_SYMBOL_MAP:
        rel_mod, attr_name = _KERNEL_SYMBOL_MAP[name]
        module = importlib.import_module(f"neuralops.ops.cuda.{rel_mod}")
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    if name in _SPECIAL_MAP:
        mod_path, attr_name = _SPECIAL_MAP[name]
        module = importlib.import_module(mod_path)
        val = getattr(module, attr_name)
        globals()[name] = val
        return val
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return __all__


__all__ = sorted(
    list(_KERNEL_SYMBOL_MAP.keys())
    + list(_SPECIAL_MAP.keys())
    + ["CUDA_KERNEL_CATALOG", "list_cuda_kernels"]
)

if "annotations" in globals():
    del annotations
