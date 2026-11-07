#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
PyTorch Reference Implementations (Tier 1)

Clean vectorized PyTorch fallback reference implementations for hardware operators.
"""

from __future__ import annotations

import importlib
from typing import Any

# Registry mapping canonical PyTorch reference names to their autograd module implementations
_PYTORCH_REF_MAP: dict[str, tuple[str, str]] = {
    # Normalization & Elementwise
    "rmsnorm_pytorch_ref": ("neuralops.ops.cuda.rmsnorm.rmsnorm_autograd", "rmsnorm_pytorch_ref"),
    "layernorm_pytorch_ref": ("neuralops.ops.cuda.layernorm.layernorm_autograd", "layernorm_pytorch_ref"),
    "fast_gelu_pytorch_ref": ("neuralops.ops.cuda.fast_gelu.fast_gelu_autograd", "fast_gelu_pytorch_ref"),
    "swiglu_pytorch_ref": ("neuralops.ops.cuda.swiglu.swiglu_autograd", "swiglu_pytorch_ref"),
    "fused_softplus_pytorch_ref": ("neuralops.ops.cuda.softplus.softplus_autograd", "fused_softplus_pytorch_ref"),
    "rms_variance_match_pytorch_ref": ("neuralops.ops.cuda.rms_variance_match.rms_variance_match_autograd", "rms_variance_match_pytorch_ref"),
    "split_head_norm_pytorch_ref": ("neuralops.ops.cuda.split_head_norm.split_head_norm_autograd", "split_head_norm_pytorch_ref"),
    "fused_norm_linear_pytorch_ref": ("neuralops.ops.cuda.fused_norm_linear.norm_linear_autograd", "fused_norm_linear_pytorch_ref"),
    "fused_dropout_add_pytorch_ref": ("neuralops.ops.cuda.fused_dropout_add.dropout_add_autograd", "fused_dropout_add_pytorch_ref"),

    # MoE Routing
    "moe_routing_pytorch_ref": ("neuralops.ops.cuda.moe_routing.moe_autograd", "moe_routing_pytorch_ref"),
    "moe_dispatch_gather_pytorch_ref": ("neuralops.ops.cuda.moe_routing.moe_autograd", "moe_dispatch_gather_pytorch_ref"),
    "moe_route_pytorch_ref": ("neuralops.ops.cuda.moe_routing.moe_autograd", "moe_route_pytorch_ref"),
    "_moe_dispatch_pytorch_ref": ("neuralops.ops.cuda.moe_routing.moe_autograd", "_moe_dispatch_pytorch_ref"),
    "_moe_gather_pytorch_ref": ("neuralops.ops.cuda.moe_routing.moe_autograd", "_moe_gather_pytorch_ref"),

    # Loss Functions
    "fused_cross_entropy_pytorch_ref": ("neuralops.ops.cuda.fused_ce.fused_ce_autograd", "fused_cross_entropy_pytorch_ref"),
    "fused_kd_loss_pytorch_ref": ("neuralops.ops.cuda.fused_kd.kd_autograd", "fused_kd_loss_pytorch_ref"),

    # Embeddings & Positional
    "fused_rotary_emb_pytorch_ref": ("neuralops.ops.cuda.fused_rope.rope_autograd", "fused_rotary_emb_pytorch_ref"),
    "unified_rope_offset_pytorch_ref": ("neuralops.ops.cuda.unified_rope_offset.unified_rope_offset_autograd", "unified_rope_offset_pytorch_ref"),
    "alibi_attention_pytorch_ref": ("neuralops.ops.cuda.alibi_attention.alibi_autograd", "alibi_attention_pytorch_ref"),

    # Scans & Recurrent Steps
    "chunked_delta_gate_scan_pytorch_ref": ("neuralops.ops.cuda.gdn_scan.gdn_autograd", "chunked_delta_gate_scan_pytorch_ref"),
    "gdn_delta_rule_pytorch_ref": ("neuralops.ops.cuda.gdn_delta_rule.gdn_delta_rule_autograd", "gdn_delta_rule_pytorch_ref"),
    "gla_scan_pytorch_ref": ("neuralops.ops.cuda.gla_scan.gla_autograd", "gla_scan_pytorch_ref"),
    "recurrent_gla_step_pytorch_ref": ("neuralops.ops.cuda.recurrent_gla_step.gla_step_autograd", "recurrent_gla_step_pytorch_ref"),
    "selective_scan_pytorch_ref": ("neuralops.ops.cuda.selective_scan.selective_scan_autograd", "selective_scan_pytorch_ref"),
    "fms_scan_pytorch_ref": ("neuralops.ops.cuda.fms_scan.fms_autograd", "fms_scan_pytorch_ref"),
    "hgrn2_fused_bwd_scan_pytorch_ref": ("neuralops.ops.cuda.hgrn2_scan.hgrn2_autograd", "hgrn2_fused_bwd_scan_pytorch_ref"),
    "recurrent_linear_step_pytorch_ref": ("neuralops.ops.cuda.state_update.state_autograd", "recurrent_linear_step_pytorch_ref"),
    "recurrent_hgrn2_step_pytorch_ref": ("neuralops.ops.cuda.recurrent_hgrn2_step.hgrn2_step_autograd", "recurrent_hgrn2_step_pytorch_ref"),

    # Attentions & KV Cache
    "linear_attention_pytorch_ref": ("neuralops.ops.cuda.linear_attention.linear_attn_autograd", "linear_attention_pytorch_ref"),
    "cosformer_linear_attention_pytorch_ref": ("neuralops.ops.cuda.cosformer.cosformer_autograd", "cosformer_linear_attention_pytorch_ref"),
    "multi_scale_retention_pytorch_ref": ("neuralops.ops.cuda.multi_scale_retention.multi_scale_retention_autograd", "multi_scale_retention_pytorch_ref"),
    "block_sparse_attn_pytorch_ref": ("neuralops.ops.cuda.sparse_attention.sparse_attention_autograd", "block_sparse_attn_pytorch_ref"),
    "sliding_window_pytorch_ref": ("neuralops.ops.cuda.sliding_window.sliding_autograd", "sliding_window_pytorch_ref"),
    "varlen_attention_pytorch_ref": ("neuralops.ops.cuda.varlen_attention.varlen_autograd", "varlen_attention_pytorch_ref"),
    "paged_attention_pytorch_reference": ("neuralops.ops.cuda.paged_attention.paged_attention_autograd", "paged_attention_pytorch_reference"),
    "flash_decoding_attention_pytorch_ref": ("neuralops.ops.cuda.flash_decoding.flash_dec_autograd", "flash_decoding_attention_pytorch_ref"),
    "latent_kv_cache_attention_pytorch_ref": ("neuralops.ops.cuda.latent_kv_attn.latent_kv_autograd", "latent_kv_cache_attention_pytorch_ref"),
    "paged_recurrent_step_pytorch_ref": ("neuralops.ops.cuda.paged_recurrent_step.paged_recurrent_step_autograd", "paged_recurrent_step_pytorch_ref"),

    # SVD & Low-Rank GEMMs
    "fused_lowrank_gated_gemm_pytorch_ref": ("neuralops.ops.cuda.low_rank_svd.lowrank_autograd", "fused_lowrank_gated_gemm_pytorch_ref"),
    "rotational_svd_pytorch_ref": ("neuralops.ops.cuda.rotational_svd.rotational_svd_autograd", "rotational_svd_pytorch_ref"),
    "kronecker_matmul_pytorch_ref": ("neuralops.ops.cuda.kronecker_gemm.kronecker_autograd", "kronecker_matmul_pytorch_ref"),
    "spectral_entropy_svd_gate_pytorch_ref": ("neuralops.ops.cuda.entropy_gate.entropy_gate_autograd", "spectral_entropy_svd_gate_pytorch_ref"),
    "svd_linear_decay_pytorch_ref": ("neuralops.ops.cuda.svd_linear_decay.svd_linear_decay_autograd", "svd_linear_decay_pytorch_ref"),
    "svd_mamba_fused_pytorch_ref": ("neuralops.ops.cuda.svd_mamba_fused.svd_mamba_fused_autograd", "svd_mamba_fused_pytorch_ref"),
    "hadamard_w4a16_gemm_pytorch_ref": ("neuralops.ops.cuda.hadamard_gemm.hadamard_autograd", "hadamard_w4a16_gemm_pytorch_ref"),
    "fp8_gemm_pytorch_ref": ("neuralops.ops.cuda.fp8_gemm.fp8_autograd", "fp8_gemm_pytorch_ref"),
    "fp4_gemm_pytorch_ref": ("neuralops.ops.cuda.fp4_gemm.fp4_autograd", "fp4_gemm_pytorch_ref"),
}

__all__ = sorted(list(_PYTORCH_REF_MAP.keys()))


def __getattr__(name: str) -> Any:
    if name in _PYTORCH_REF_MAP:
        module_path, attr_name = _PYTORCH_REF_MAP[name]
        try:
            mod = importlib.import_module(module_path)
            val = getattr(mod, attr_name)
            globals()[name] = val
            return val
        except (ImportError, AttributeError) as exc:
            raise ImportError(f"Could not load PyTorch reference '{name}' from '{module_path}': {exc}") from exc
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    return sorted(list(set(globals().keys()) | set(__all__)))
