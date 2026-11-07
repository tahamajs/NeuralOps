#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps CUDA Hardware Operators.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any, List, Optional

from neuralops.core.cuda_kernels import act_scale_gemm, act_scale_quant_gemm, act_scale_quant_gemm_pytorch_ref
from neuralops.core.cuda_kernels import adaptive_span
from neuralops.core.cuda_kernels import adaptive_span as adaptive_span_op
from neuralops.core.cuda_kernels import aegis_gate_norm
from neuralops.core.cuda_kernels import aegis_gate_norm as aegis_gate_norm_op
from neuralops.core.cuda_kernels import aegis_gate_norm_pytorch_ref
from neuralops.core.cuda_kernels import alibi_attention
from neuralops.core.cuda_kernels import alibi_attention as alibi_attention_op
from neuralops.core.cuda_kernels import alibi_attention_pytorch_ref
from neuralops.core.cuda_kernels import bigbird
from neuralops.core.cuda_kernels import bigbird as bigbird_op
from neuralops.core.cuda_kernels import bigbird_ref
from neuralops.core.cuda_kernels import bitonic_topk
from neuralops.core.cuda_kernels import bitonic_topk as bitonic_topk_op
from neuralops.core.cuda_kernels import (
    bitonic_topk_pytorch_ref,
    block_diag_gemm,
    block_diag_matmul,
    block_diag_matmul_pytorch_ref,
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)
from neuralops.core.cuda_kernels import causal_conv1d
from neuralops.core.cuda_kernels import causal_conv1d as causal_conv1d_op
from neuralops.core.cuda_kernels import causal_conv1d_pytorch_ref
from neuralops.core.cuda_kernels import channel_shuffle
from neuralops.core.cuda_kernels import channel_shuffle as channel_shuffle_op
from neuralops.core.cuda_kernels import (
    channel_shuffle_pytorch_ref,
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels import cluster_attention
from neuralops.core.cuda_kernels import cluster_attention as cluster_attention_op
from neuralops.core.cuda_kernels import (
    cluster_attention_ref,
    copy_blocks_cow,
    copy_blocks_cow_pytorch_ref,
    cosformer,
    cosformer_linear_attention,
    cosformer_linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels import cross_attention
from neuralops.core.cuda_kernels import cross_attention as cross_attention_op
from neuralops.core.cuda_kernels import cross_attention_pytorch_ref
from neuralops.core.cuda_kernels import decay_linear_attention
from neuralops.core.cuda_kernels import decay_linear_attention as decay_linear_attention_op
from neuralops.core.cuda_kernels import decay_linear_attention_ref
from neuralops.core.cuda_kernels import differential_attention
from neuralops.core.cuda_kernels import differential_attention as differential_attention_op
from neuralops.core.cuda_kernels import differential_attention_ref
from neuralops.core.cuda_kernels import dynamic_conv1d
from neuralops.core.cuda_kernels import dynamic_conv1d as dynamic_conv1d_op
from neuralops.core.cuda_kernels import entmax_bisect
from neuralops.core.cuda_kernels import entmax_bisect as entmax_bisect_op
from neuralops.core.cuda_kernels import entropy_gate
from neuralops.core.cuda_kernels import entropy_pruning
from neuralops.core.cuda_kernels import entropy_pruning as entropy_pruning_op
from neuralops.core.cuda_kernels import entropy_pruning_pytorch_ref
from neuralops.core.cuda_kernels import fast_attention
from neuralops.core.cuda_kernels import fast_attention as fast_attention_op
from neuralops.core.cuda_kernels import fast_attention_ref
from neuralops.core.cuda_kernels import fast_gelu
from neuralops.core.cuda_kernels import fast_gelu as fast_gelu_op
from neuralops.core.cuda_kernels import fast_gelu_pytorch_ref, fast_hadamard_transform
from neuralops.core.cuda_kernels import favor_plus
from neuralops.core.cuda_kernels import favor_plus as favor_plus_op
from neuralops.core.cuda_kernels import favor_plus_pytorch_ref
from neuralops.core.cuda_kernels import flash_decode_stage2
from neuralops.core.cuda_kernels import flash_decode_stage2 as flash_decode_stage2_op
from neuralops.core.cuda_kernels import (
    flash_decode_stage2_pytorch_ref,
    flash_decoding,
    flash_decoding_attention,
    flash_decoding_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels import flash_gqa
from neuralops.core.cuda_kernels import flash_gqa as flash_gqa_op
from neuralops.core.cuda_kernels import flash_gqa_pytorch_ref
from neuralops.core.cuda_kernels import fms_scan
from neuralops.core.cuda_kernels import fms_scan as fms_scan_op
from neuralops.core.cuda_kernels import fms_scan_pytorch_ref
from neuralops.core.cuda_kernels import fp4_gemm
from neuralops.core.cuda_kernels import fp4_gemm as fp4_gemm_op
from neuralops.core.cuda_kernels import fp4_gemm_pytorch_ref
from neuralops.core.cuda_kernels import fp8_gemm
from neuralops.core.cuda_kernels import fp8_gemm as fp8_gemm_op
from neuralops.core.cuda_kernels import (
    fp8_gemm_pytorch_ref,
    fused_adaptive_span_penalty_pytorch,
    fused_ce,
    fused_cross_entropy,
    fused_cross_entropy_pytorch_ref,
    fused_dropout_add,
    fused_dropout_add_pytorch_ref,
    fused_dynamic_conv1d_pytorch,
    fused_entmax_bisect_pytorch,
    fused_kd,
    fused_kd_loss,
    fused_kd_loss_pytorch_ref,
    fused_lowrank_gated_gemm,
    fused_lowrank_gated_gemm_pytorch_ref,
    fused_norm_linear,
    fused_norm_linear_pytorch_ref,
    fused_rms_variance_match_triton,
    fused_rope,
    fused_rotary_emb,
    fused_rotary_emb_pytorch_ref,
    fused_softplus,
    fused_softplus_pytorch_ref,
    fused_unified_rope_offset_pytorch,
)
from neuralops.core.cuda_kernels import fwht_transform
from neuralops.core.cuda_kernels import fwht_transform as fwht_transform_op
from neuralops.core.cuda_kernels import galerkin_attention
from neuralops.core.cuda_kernels import galerkin_attention as galerkin_attention_op
from neuralops.core.cuda_kernels import galerkin_attention_ref, gated_deltanet_block
from neuralops.core.cuda_kernels import gdn_delta_rule
from neuralops.core.cuda_kernels import gdn_delta_rule as gdn_delta_rule_op
from neuralops.core.cuda_kernels import gdn_delta_rule_pytorch_ref, gdn_scan
from neuralops.core.cuda_kernels import gla_scan
from neuralops.core.cuda_kernels import gla_scan as gla_scan_op
from neuralops.core.cuda_kernels import gla_scan_pytorch_ref
from neuralops.core.cuda_kernels import gla_unnormalized_scan
from neuralops.core.cuda_kernels import gla_unnormalized_scan as gla_unnormalized_scan_op
from neuralops.core.cuda_kernels import gla_unnormalized_scan_pytorch_ref
from neuralops.core.cuda_kernels import goldfinch_ahn
from neuralops.core.cuda_kernels import goldfinch_ahn as goldfinch_ahn_op
from neuralops.core.cuda_kernels import goldfinch_ahn_ref
from neuralops.core.cuda_kernels import griffin_ssd
from neuralops.core.cuda_kernels import griffin_ssd as griffin_ssd_op
from neuralops.core.cuda_kernels import (
    griffin_ssd_ref,
    hadamard_gemm,
    hadamard_w4a16_gemm,
    hadamard_w4a16_gemm_pytorch_ref,
    hgrn2_fused_bwd_scan,
    hgrn2_fused_bwd_scan_pytorch_ref,
    hgrn2_gated_deltanet,
    hgrn2_gated_outer,
    hgrn2_layer,
    hgrn2_scan,
)
from neuralops.core.cuda_kernels import hgrn2_spatial_scan
from neuralops.core.cuda_kernels import hgrn2_spatial_scan as hgrn2_spatial_scan_op
from neuralops.core.cuda_kernels import hgrn2_spatial_scan_pytorch_ref
from neuralops.core.cuda_kernels import hstu_attention
from neuralops.core.cuda_kernels import hstu_attention as hstu_attention_op
from neuralops.core.cuda_kernels import hstu_attention_ref
from neuralops.core.cuda_kernels import hydra_attention
from neuralops.core.cuda_kernels import hydra_attention as hydra_attention_op
from neuralops.core.cuda_kernels import hydra_attention_ref
from neuralops.core.cuda_kernels import hymba_transx
from neuralops.core.cuda_kernels import hymba_transx as hymba_transx_op
from neuralops.core.cuda_kernels import hymba_transx_ref
from neuralops.core.cuda_kernels import jamba
from neuralops.core.cuda_kernels import jamba as jamba_op
from neuralops.core.cuda_kernels import jamba_info
from neuralops.core.cuda_kernels import jamba_info as jamba_info_op
from neuralops.core.cuda_kernels import (
    jamba_info_ref,
    jamba_ref,
    kronecker_gemm,
    kronecker_matmul,
    kronecker_matmul_pytorch_ref,
    latent_kv_attn,
    latent_kv_cache_attention,
    latent_kv_cache_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels import layernorm
from neuralops.core.cuda_kernels import layernorm as layernorm_op
from neuralops.core.cuda_kernels import layernorm_pytorch_ref
from neuralops.core.cuda_kernels import linear_attention
from neuralops.core.cuda_kernels import linear_attention as linear_attention_op
from neuralops.core.cuda_kernels import linear_attention_pytorch_ref
from neuralops.core.cuda_kernels import linear_rnn_attention
from neuralops.core.cuda_kernels import linear_rnn_attention as linear_rnn_attention_op
from neuralops.core.cuda_kernels import linear_rnn_attention_ref
from neuralops.core.cuda_kernels import linformer_attention
from neuralops.core.cuda_kernels import linformer_attention as linformer_attention_op
from neuralops.core.cuda_kernels import linformer_attention_ref
from neuralops.core.cuda_kernels import liquid_attention
from neuralops.core.cuda_kernels import liquid_attention as liquid_attention_op
from neuralops.core.cuda_kernels import liquid_attention_ref
from neuralops.core.cuda_kernels import local_attention
from neuralops.core.cuda_kernels import local_attention as local_attention_op
from neuralops.core.cuda_kernels import local_attention_ref
from neuralops.core.cuda_kernels import logcumsumexp
from neuralops.core.cuda_kernels import logcumsumexp as logcumsumexp_op
from neuralops.core.cuda_kernels import logcumsumexp_pytorch_ref
from neuralops.core.cuda_kernels import longformer
from neuralops.core.cuda_kernels import longformer as longformer_op
from neuralops.core.cuda_kernels import (
    longformer_ref,
    low_rank_svd,
    lut_codebook_dequant_gemm,
    lut_codebook_dequant_gemm_pytorch_ref,
    lut_gemm,
)
from neuralops.core.cuda_kernels import mamba2_block
from neuralops.core.cuda_kernels import mamba2_block as mamba2_block_op
from neuralops.core.cuda_kernels import mamba2_block_ref
from neuralops.core.cuda_kernels import mamba_block
from neuralops.core.cuda_kernels import mamba_block as mamba_block_op
from neuralops.core.cuda_kernels import mamba_block_ref
from neuralops.core.cuda_kernels import mamba_svd_flash
from neuralops.core.cuda_kernels import mamba_svd_flash as mamba_svd_flash_op
from neuralops.core.cuda_kernels import mamba_svd_flash_ref, moe_dispatch_gather, moe_route_pytorch_ref, moe_routing
from neuralops.core.cuda_kernels import multi_scale_retention
from neuralops.core.cuda_kernels import multi_scale_retention as multi_scale_retention_op
from neuralops.core.cuda_kernels import multi_scale_retention_pytorch_ref
from neuralops.core.cuda_kernels import nasc_attention
from neuralops.core.cuda_kernels import nasc_attention as nasc_attention_op
from neuralops.core.cuda_kernels import nasc_attention_ref
from neuralops.core.cuda_kernels import nexus
from neuralops.core.cuda_kernels import nexus as nexus_op
from neuralops.core.cuda_kernels import nexus_ref
from neuralops.core.cuda_kernels import nystrom_attention
from neuralops.core.cuda_kernels import nystrom_attention as nystrom_attention_op
from neuralops.core.cuda_kernels import nystrom_attention_ref
from neuralops.core.cuda_kernels import orion
from neuralops.core.cuda_kernels import orion as orion_op
from neuralops.core.cuda_kernels import orion_ref
from neuralops.core.cuda_kernels import paged_attention
from neuralops.core.cuda_kernels import paged_attention as paged_attention_op
from neuralops.core.cuda_kernels import paged_attention_pytorch_reference, paged_kv_cache, paged_kv_compact_pytorch_ref
from neuralops.core.cuda_kernels import paged_kv_evict
from neuralops.core.cuda_kernels import paged_kv_evict as paged_kv_evict_op
from neuralops.core.cuda_kernels import paged_recurrent_state_step_pytorch
from neuralops.core.cuda_kernels import paged_recurrent_step
from neuralops.core.cuda_kernels import paged_recurrent_step as paged_recurrent_step_op
from neuralops.core.cuda_kernels import phantom
from neuralops.core.cuda_kernels import phantom as phantom_op
from neuralops.core.cuda_kernels import phantom_ref, quantized_gemm
from neuralops.core.cuda_kernels import recurrent_gla_step
from neuralops.core.cuda_kernels import recurrent_gla_step as recurrent_gla_step_op
from neuralops.core.cuda_kernels import recurrent_gla_step_pytorch_ref
from neuralops.core.cuda_kernels import recurrent_hgrn2_step
from neuralops.core.cuda_kernels import recurrent_hgrn2_step as recurrent_hgrn2_step_op
from neuralops.core.cuda_kernels import (
    recurrent_hgrn2_step_pytorch_ref,
    recurrent_linear_step,
    recurrent_linear_step_pytorch_ref,
)
from neuralops.core.cuda_kernels import reformer
from neuralops.core.cuda_kernels import reformer as reformer_op
from neuralops.core.cuda_kernels import reformer_ref, retnet_retention, retnet_retention_pytorch_ref
from neuralops.core.cuda_kernels import rfa_attention
from neuralops.core.cuda_kernels import rfa_attention as rfa_attention_op
from neuralops.core.cuda_kernels import rfa_attention_ref
from neuralops.core.cuda_kernels import rms_variance_match
from neuralops.core.cuda_kernels import rms_variance_match as rms_variance_match_op
from neuralops.core.cuda_kernels import rmsnorm
from neuralops.core.cuda_kernels import rmsnorm as rmsnorm_op
from neuralops.core.cuda_kernels import rmsnorm_pytorch_ref
from neuralops.core.cuda_kernels import rotational_svd
from neuralops.core.cuda_kernels import rotational_svd as rotational_svd_op
from neuralops.core.cuda_kernels import rotational_svd_pytorch_ref
from neuralops.core.cuda_kernels import routing_attention
from neuralops.core.cuda_kernels import routing_attention as routing_attention_op
from neuralops.core.cuda_kernels import (
    routing_attention_ref,
    rpca_sparse_svd,
    rpca_sparse_svd_gemm,
    rpca_sparse_svd_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels import rwkv6
from neuralops.core.cuda_kernels import rwkv6 as rwkv6_op
from neuralops.core.cuda_kernels import rwkv6_ref
from neuralops.core.cuda_kernels import samba_m3
from neuralops.core.cuda_kernels import samba_m3 as samba_m3_op
from neuralops.core.cuda_kernels import samba_m3_ref
from neuralops.core.cuda_kernels import samba_moe
from neuralops.core.cuda_kernels import samba_moe as samba_moe_op
from neuralops.core.cuda_kernels import samba_moe_ref
from neuralops.core.cuda_kernels import selective_scan
from neuralops.core.cuda_kernels import selective_scan as selective_scan_op
from neuralops.core.cuda_kernels import selective_scan_pytorch_ref
from neuralops.core.cuda_kernels import selective_ssm
from neuralops.core.cuda_kernels import selective_ssm as selective_ssm_op
from neuralops.core.cuda_kernels import selective_ssm_ref
from neuralops.core.cuda_kernels import simple_ssm
from neuralops.core.cuda_kernels import simple_ssm as simple_ssm_op
from neuralops.core.cuda_kernels import simple_ssm_ref, sliding_window, sliding_window_attn, sliding_window_pytorch_ref
from neuralops.core.cuda_kernels import soa_attn
from neuralops.core.cuda_kernels import soa_attn as soa_attn_op
from neuralops.core.cuda_kernels import soa_attn_ref, softplus, sparse_attention
from neuralops.core.cuda_kernels import spectra_v2
from neuralops.core.cuda_kernels import spectra_v2 as spectra_v2_op
from neuralops.core.cuda_kernels import spectra_v2_ref
from neuralops.core.cuda_kernels import spectraformer as spectraformer_op
from neuralops.core.cuda_kernels import (
    spectraformer_attention,
    spectraformer_ref,
    spectral_entropy_svd_gate,
    spectral_entropy_svd_gate_pytorch_ref,
)
from neuralops.core.cuda_kernels import spectral_mamba
from neuralops.core.cuda_kernels import spectral_mamba as spectral_mamba_op
from neuralops.core.cuda_kernels import spectral_mamba_ref
from neuralops.core.cuda_kernels import spectral_shrinkage
from neuralops.core.cuda_kernels import spectral_shrinkage as spectral_shrinkage_op
from neuralops.core.cuda_kernels import spectral_shrinkage_pytorch_ref
from neuralops.core.cuda_kernels import speculative_verify
from neuralops.core.cuda_kernels import speculative_verify as speculative_verify_op
from neuralops.core.cuda_kernels import speculative_verify_pytorch_ref
from neuralops.core.cuda_kernels import split_head_norm
from neuralops.core.cuda_kernels import split_head_norm as split_head_norm_op
from neuralops.core.cuda_kernels import split_head_norm_pytorch_ref, state_update
from neuralops.core.cuda_kernels import svd_prefix_scan
from neuralops.core.cuda_kernels import svd_prefix_scan as svd_prefix_scan_op
from neuralops.core.cuda_kernels import svd_prefix_scan_pytorch_ref
from neuralops.core.cuda_kernels import swiglu
from neuralops.core.cuda_kernels import swiglu as swiglu_op
from neuralops.core.cuda_kernels import swiglu_pytorch_ref
from neuralops.core.cuda_kernels import ternary_gemm
from neuralops.core.cuda_kernels import ternary_gemm as ternary_gemm_op
from neuralops.core.cuda_kernels import ternary_gemm_pytorch_ref
from neuralops.core.cuda_kernels import tpa_attention
from neuralops.core.cuda_kernels import tpa_attention as tpa_attention_op
from neuralops.core.cuda_kernels import tpa_attention_ref
from neuralops.core.cuda_kernels import tt_contraction
from neuralops.core.cuda_kernels import tt_contraction as tt_contraction_op
from neuralops.core.cuda_kernels import tt_contraction_pytorch_ref, tt_linear, tt_linear_pytorch_ref
from neuralops.core.cuda_kernels import unified_rope_offset
from neuralops.core.cuda_kernels import unified_rope_offset as unified_rope_offset_op
from neuralops.core.cuda_kernels import upi_stuffed_mamba
from neuralops.core.cuda_kernels import upi_stuffed_mamba as upi_stuffed_mamba_op
from neuralops.core.cuda_kernels import upi_stuffed_mamba_ref
from neuralops.core.cuda_kernels import varlen_attention
from neuralops.core.cuda_kernels import varlen_attention as varlen_attention_op
from neuralops.core.cuda_kernels import varlen_attention_pytorch_ref
from neuralops.core.cuda_kernels import vspec
from neuralops.core.cuda_kernels import vspec as vspec_op
from neuralops.core.cuda_kernels import vspec_ref, w4a16_gemm, w4a16_gemm_pytorch_ref
from neuralops.core.cuda_kernels._bindings import get_cuda_arch_flags, load_cuda_module
from neuralops.core.cuda_kernels._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    get_last_dispatch_backend,
    get_last_dispatch_error,
    reset_dispatch_stats,
)
from neuralops.core.cuda_kernels._utils import check_cuda_error, get_device_capability, is_cuda_available
from neuralops.ops.cuda._map import _KERNEL_SYMBOL_MAP, CUDA_KERNEL_CATALOG

CATALOG = CUDA_KERNEL_CATALOG


def list_cuda_kernels(category: Optional[str] = None) -> List[str]:
    """Return list of all available CUDA kernels in NeuralOps.

    Args:
        category: Filter category:
            - None or 'all': All native CUDA kernel implementations (.cu files / catalog entries).
            - 'modules': Unique module package names.
            - 'operators': Primary CUDA operator function names.

    Returns:
        Sorted list of kernel names.
    """
    if category == "modules":
        return sorted(list({v[0] for v in _KERNEL_SYMBOL_MAP.values() if not v[0].startswith("_")}))
    if category == "operators":
        import neuralops.core.cuda_kernels as nck

        util_names = {
            "load_cuda_module",
            "get_cuda_arch_flags",
            "dispatch",
            "force_kernel_backend",
            "get_dispatch_stats",
            "reset_dispatch_stats",
            "is_cuda_available",
            "get_device_capability",
            "check_cuda_error",
            "list_cuda_kernels",
        }
        return sorted(
            [
                k
                for k in nck.__all__
                if k not in util_names
                and not k.endswith("_pytorch_ref")
                and not k.endswith("_pytorch_reference")
                and not k.endswith("_pytorch")
                and not k.endswith("_ref")
            ]
        )
    return list(CUDA_KERNEL_CATALOG)


__all__ = [
    "load_cuda_module",
    "get_cuda_arch_flags",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "get_last_dispatch_backend",
    "get_last_dispatch_error",
    "is_cuda_available",
    "get_device_capability",
    "check_cuda_error",
    "CATALOG",
    "CUDA_KERNEL_CATALOG",
    "list_cuda_kernels",
    "act_scale_gemm",
    "adaptive_span",
    "aegis_gate_norm",
    "alibi_attention",
    "bigbird",
    "bitonic_topk",
    "block_diag_gemm",
    "causal_conv1d",
    "channel_shuffle",
    "cluster_attention",
    "cosformer",
    "cross_attention",
    "decay_linear_attention",
    "differential_attention",
    "dynamic_conv1d",
    "entmax_bisect",
    "entropy_gate",
    "entropy_pruning",
    "fast_attention",
    "fast_gelu",
    "favor_plus",
    "flash_decode_stage2",
    "flash_decoding",
    "flash_gqa",
    "fms_scan",
    "fp4_gemm",
    "fp8_gemm",
    "fused_ce",
    "fused_dropout_add",
    "fused_kd",
    "fused_norm_linear",
    "fused_rope",
    "fwht_transform",
    "galerkin_attention",
    "gdn_delta_rule",
    "gdn_scan",
    "gla_scan",
    "gla_unnormalized_scan",
    "goldfinch_ahn",
    "griffin_ssd",
    "hadamard_gemm",
    "hgrn2_scan",
    "hgrn2_spatial_scan",
    "hstu_attention",
    "hydra_attention",
    "hymba_transx",
    "jamba",
    "jamba_info",
    "kronecker_gemm",
    "latent_kv_attn",
    "layernorm",
    "linear_attention",
    "linear_rnn_attention",
    "linformer_attention",
    "liquid_attention",
    "local_attention",
    "logcumsumexp",
    "longformer",
    "low_rank_svd",
    "lut_gemm",
    "mamba2_block",
    "mamba_block",
    "mamba_svd_flash",
    "moe_routing",
    "multi_scale_retention",
    "nasc_attention",
    "nexus",
    "nystrom_attention",
    "orion",
    "paged_attention",
    "paged_kv_cache",
    "paged_kv_evict",
    "paged_recurrent_step",
    "phantom",
    "quantized_gemm",
    "recurrent_gla_step",
    "recurrent_hgrn2_step",
    "reformer",
    "rfa_attention",
    "rms_variance_match",
    "rmsnorm",
    "rotational_svd",
    "routing_attention",
    "rpca_sparse_svd",
    "rwkv6",
    "samba_m3",
    "samba_moe",
    "selective_scan",
    "selective_ssm",
    "simple_ssm",
    "sliding_window",
    "soa_attn",
    "softplus",
    "sparse_attention",
    "spectra_v2",
    "spectraformer_attention",
    "spectral_mamba",
    "spectral_shrinkage",
    "speculative_verify",
    "split_head_norm",
    "state_update",
    "svd_prefix_scan",
    "swiglu",
    "ternary_gemm",
    "tpa_attention",
    "tt_contraction",
    "unified_rope_offset",
    "upi_stuffed_mamba",
    "varlen_attention",
    "vspec",
    "fused_lowrank_gated_gemm",
    "fused_lowrank_gated_gemm_pytorch_ref",
    "chunked_delta_gate_scan",
    "chunked_delta_gate_scan_pytorch_ref",
    "hgrn2_fused_bwd_scan",
    "hgrn2_fused_bwd_scan_pytorch_ref",
    "cosformer_linear_attention",
    "cosformer_linear_attention_pytorch_ref",
    "fused_rotary_emb",
    "fused_rotary_emb_pytorch_ref",
    "kronecker_matmul",
    "kronecker_matmul_pytorch_ref",
    "lut_codebook_dequant_gemm",
    "lut_codebook_dequant_gemm_pytorch_ref",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch_ref",
    "rpca_sparse_svd_gemm",
    "rpca_sparse_svd_gemm_pytorch_ref",
    "tt_linear",
    "tt_linear_pytorch_ref",
    "hadamard_w4a16_gemm",
    "hadamard_w4a16_gemm_pytorch_ref",
    "act_scale_quant_gemm",
    "act_scale_quant_gemm_pytorch_ref",
    "latent_kv_cache_attention",
    "latent_kv_cache_attention_pytorch_ref",
    "spectral_entropy_svd_gate",
    "spectral_entropy_svd_gate_pytorch_ref",
    "fused_kd_loss",
    "fused_kd_loss_pytorch_ref",
    "recurrent_linear_step",
    "recurrent_linear_step_pytorch_ref",
    "flash_decoding_attention",
    "flash_decoding_attention_pytorch_ref",
    "fused_softplus",
    "fused_softplus_pytorch_ref",
    "fused_dropout_add",
    "fused_dropout_add_pytorch_ref",
    "block_diag_matmul",
    "block_diag_matmul_pytorch_ref",
    "sliding_window_attn",
    "sliding_window_pytorch_ref",
    "block_sparse_attn",
    "block_sparse_attn_pytorch_ref",
    "fast_hadamard_transform",
    "fused_rms_variance_match_triton",
    "fused_adaptive_span_penalty_pytorch",
    "fused_dynamic_conv1d_pytorch",
    "paged_recurrent_state_step_pytorch",
    "paged_kv_compact_pytorch_ref",
    "fused_unified_rope_offset_pytorch",
    "fused_entmax_bisect_pytorch",
    "retnet_retention",
    "retnet_retention_pytorch_ref",
    "hgrn2_gated_deltanet",
    "gated_deltanet_block",
    "hgrn2_layer",
    "hgrn2_gated_outer",
]


def __getattr__(name: str) -> Any:
    """Dynamically resolve attributes from neuralops.core.cuda_kernels."""
    import neuralops.core.cuda_kernels as nck

    if hasattr(nck, name):
        val = getattr(nck, name)
        globals()[name] = val
        return val
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> list[str]:
    """Execute __dir__ operation."""
    return sorted(list(set(globals().keys()) | set(__all__)))
