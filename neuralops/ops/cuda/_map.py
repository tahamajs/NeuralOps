#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Symbol mapping registry for lazy loading native CUDA kernels.
"""

from __future__ import annotations

_SPECIAL_MAP: dict[str, tuple[str, str]] = {
    "hgrn2_gated_deltanet": ("neuralops.layers.gated.hgrn2_gated_deltanet", "HGRN2GatedDeltaNet"),
    "gated_deltanet_block": ("neuralops.layers.gated.gated_deltanet_block", "GatedDeltaNetBlock"),
    "hgrn2_layer": ("neuralops.layers.gated.hgrn2_gated_layer", "HGRN2Layer"),
    "hgrn2_gated_outer": ("neuralops.layers.gated.hgrn2_gated_outer", "HGRN2GatedOuterBlock"),
}

_KERNEL_SYMBOL_MAP: dict[str, tuple[str, str]] = {
  "load_cuda_module": [
    "_bindings",
    "load_cuda_module"
  ],
  "get_cuda_arch_flags": [
    "_bindings",
    "get_cuda_arch_flags"
  ],
  "dispatch": [
    "_fallback",
    "dispatch"
  ],
  "force_kernel_backend": [
    "_fallback",
    "force_kernel_backend"
  ],
  "get_dispatch_stats": [
    "_fallback",
    "get_dispatch_stats"
  ],
  "reset_dispatch_stats": [
    "_fallback",
    "reset_dispatch_stats"
  ],
  "is_cuda_available": [
    "_utils",
    "is_cuda_available"
  ],
  "get_device_capability": [
    "_utils",
    "get_device_capability"
  ],
  "check_cuda_error": [
    "_utils",
    "check_cuda_error"
  ],
  "fused_lowrank_gated_gemm": [
    "low_rank_svd",
    "fused_lowrank_gated_gemm"
  ],
  "fused_lowrank_gated_gemm_pytorch_ref": [
    "low_rank_svd",
    "fused_lowrank_gated_gemm_pytorch_ref"
  ],
  "chunked_delta_gate_scan": [
    "gdn_scan",
    "chunked_delta_gate_scan"
  ],
  "chunked_delta_gate_scan_pytorch_ref": [
    "gdn_scan",
    "chunked_delta_gate_scan_pytorch_ref"
  ],
  "hgrn2_fused_bwd_scan": [
    "hgrn2_scan",
    "hgrn2_fused_bwd_scan"
  ],
  "hgrn2_fused_bwd_scan_pytorch_ref": [
    "hgrn2_scan",
    "hgrn2_fused_bwd_scan_pytorch_ref"
  ],
  "cosformer_linear_attention": [
    "cosformer",
    "cosformer_linear_attention"
  ],
  "cosformer_linear_attention_pytorch_ref": [
    "cosformer",
    "cosformer_linear_attention_pytorch_ref"
  ],
  "fused_norm_linear": [
    "fused_norm_linear",
    "fused_norm_linear"
  ],
  "fused_norm_linear_pytorch_ref": [
    "fused_norm_linear",
    "fused_norm_linear_pytorch_ref"
  ],
  "fused_rotary_emb": [
    "fused_rope",
    "fused_rotary_emb"
  ],
  "fused_rotary_emb_pytorch_ref": [
    "fused_rope",
    "fused_rotary_emb_pytorch_ref"
  ],
  "kronecker_matmul": [
    "kronecker_gemm",
    "kronecker_matmul"
  ],
  "kronecker_matmul_pytorch_ref": [
    "kronecker_gemm",
    "kronecker_matmul_pytorch_ref"
  ],
  "lut_codebook_dequant_gemm": [
    "lut_gemm",
    "lut_codebook_dequant_gemm"
  ],
  "lut_codebook_dequant_gemm_pytorch_ref": [
    "lut_gemm",
    "lut_codebook_dequant_gemm_pytorch_ref"
  ],
  "linear_attention": [
    "linear_attention",
    "linear_attention"
  ],
  "linear_attention_pytorch_ref": [
    "linear_attention",
    "linear_attention_pytorch_ref"
  ],
  "fused_cross_entropy": [
    "fused_ce",
    "fused_cross_entropy"
  ],
  "fused_cross_entropy_pytorch_ref": [
    "fused_ce",
    "fused_cross_entropy_pytorch_ref"
  ],
  "rpca_sparse_svd_gemm": [
    "rpca_sparse_svd",
    "rpca_sparse_svd_gemm"
  ],
  "rpca_sparse_svd_gemm_pytorch_ref": [
    "rpca_sparse_svd",
    "rpca_sparse_svd_gemm_pytorch_ref"
  ],
  "tt_contraction": [
    "tt_contraction",
    "tt_contraction"
  ],
  "tt_contraction_pytorch_ref": [
    "tt_contraction",
    "tt_contraction_pytorch_ref"
  ],
  "tt_linear": [
    "tt_contraction",
    "tt_linear"
  ],
  "tt_linear_pytorch_ref": [
    "tt_contraction",
    "tt_linear_pytorch_ref"
  ],
  "hadamard_w4a16_gemm": [
    "hadamard_gemm",
    "hadamard_w4a16_gemm"
  ],
  "hadamard_w4a16_gemm_pytorch_ref": [
    "hadamard_gemm",
    "hadamard_w4a16_gemm_pytorch_ref"
  ],
  "act_scale_quant_gemm": [
    "act_scale_gemm",
    "act_scale_quant_gemm"
  ],
  "act_scale_quant_gemm_pytorch_ref": [
    "act_scale_gemm",
    "act_scale_quant_gemm_pytorch_ref"
  ],
  "latent_kv_cache_attention": [
    "latent_kv_attn",
    "latent_kv_cache_attention"
  ],
  "latent_kv_cache_attention_pytorch_ref": [
    "latent_kv_attn",
    "latent_kv_cache_attention_pytorch_ref"
  ],
  "spectral_entropy_svd_gate": [
    "entropy_gate",
    "spectral_entropy_svd_gate"
  ],
  "spectral_entropy_svd_gate_pytorch_ref": [
    "entropy_gate",
    "spectral_entropy_svd_gate_pytorch_ref"
  ],
  "fused_kd_loss": [
    "fused_kd",
    "fused_kd_loss"
  ],
  "fused_kd_loss_pytorch_ref": [
    "fused_kd",
    "fused_kd_loss_pytorch_ref"
  ],
  "recurrent_linear_step": [
    "state_update",
    "recurrent_linear_step"
  ],
  "recurrent_linear_step_pytorch_ref": [
    "state_update",
    "recurrent_linear_step_pytorch_ref"
  ],
  "flash_decoding_attention": [
    "flash_decoding",
    "flash_decoding_attention"
  ],
  "flash_decoding_attention_pytorch_ref": [
    "flash_decoding",
    "flash_decoding_attention_pytorch_ref"
  ],
  "fms_scan": [
    "fms_scan",
    "fms_scan"
  ],
  "fms_scan_pytorch_ref": [
    "fms_scan",
    "fms_scan_pytorch_ref"
  ],
  "rmsnorm": [
    "rmsnorm",
    "rmsnorm"
  ],
  "rmsnorm_pytorch_ref": [
    "rmsnorm",
    "rmsnorm_pytorch_ref"
  ],
  "layernorm": [
    "layernorm",
    "layernorm"
  ],
  "layernorm_pytorch_ref": [
    "layernorm",
    "layernorm_pytorch_ref"
  ],
  "swiglu": [
    "swiglu",
    "swiglu"
  ],
  "swiglu_pytorch_ref": [
    "swiglu",
    "swiglu_pytorch_ref"
  ],
  "fast_gelu": [
    "fast_gelu",
    "fast_gelu"
  ],
  "fast_gelu_pytorch_ref": [
    "fast_gelu",
    "fast_gelu_pytorch_ref"
  ],
  "fused_softplus": [
    "softplus",
    "fused_softplus"
  ],
  "fused_softplus_pytorch_ref": [
    "softplus",
    "fused_softplus_pytorch_ref"
  ],
  "gla_scan": [
    "gla_scan",
    "gla_scan"
  ],
  "gla_scan_pytorch_ref": [
    "gla_scan",
    "gla_scan_pytorch_ref"
  ],
  "causal_conv1d": [
    "causal_conv1d",
    "causal_conv1d"
  ],
  "causal_conv1d_pytorch_ref": [
    "causal_conv1d",
    "causal_conv1d_pytorch_ref"
  ],
  "recurrent_gla_step": [
    "recurrent_gla_step",
    "recurrent_gla_step"
  ],
  "recurrent_gla_step_pytorch_ref": [
    "recurrent_gla_step",
    "recurrent_gla_step_pytorch_ref"
  ],
  "recurrent_hgrn2_step": [
    "recurrent_hgrn2_step",
    "recurrent_hgrn2_step"
  ],
  "recurrent_hgrn2_step_pytorch_ref": [
    "recurrent_hgrn2_step",
    "recurrent_hgrn2_step_pytorch_ref"
  ],
  "fused_dropout_add": [
    "fused_dropout_add",
    "fused_dropout_add"
  ],
  "fused_dropout_add_pytorch_ref": [
    "fused_dropout_add",
    "fused_dropout_add_pytorch_ref"
  ],
  "varlen_attention": [
    "varlen_attention",
    "varlen_attention"
  ],
  "varlen_attention_pytorch_ref": [
    "varlen_attention",
    "varlen_attention_pytorch_ref"
  ],
  "cross_attention": [
    "cross_attention",
    "cross_attention"
  ],
  "cross_attention_pytorch_ref": [
    "cross_attention",
    "cross_attention_pytorch_ref"
  ],
  "rotational_svd": [
    "rotational_svd",
    "rotational_svd"
  ],
  "rotational_svd_pytorch_ref": [
    "rotational_svd",
    "rotational_svd_pytorch_ref"
  ],
  "block_diag_matmul": [
    "block_diag_gemm",
    "block_diag_matmul"
  ],
  "block_diag_matmul_pytorch_ref": [
    "block_diag_gemm",
    "block_diag_matmul_pytorch_ref"
  ],
  "ternary_gemm": [
    "ternary_gemm",
    "ternary_gemm"
  ],
  "ternary_gemm_pytorch_ref": [
    "ternary_gemm",
    "ternary_gemm_pytorch_ref"
  ],
  "fp4_gemm": [
    "fp4_gemm",
    "fp4_gemm"
  ],
  "fp4_gemm_pytorch_ref": [
    "fp4_gemm",
    "fp4_gemm_pytorch_ref"
  ],
  "channel_shuffle": [
    "channel_shuffle",
    "channel_shuffle"
  ],
  "channel_shuffle_pytorch_ref": [
    "channel_shuffle",
    "channel_shuffle_pytorch_ref"
  ],
  "speculative_verify": [
    "speculative_verify",
    "speculative_verify"
  ],
  "speculative_verify_pytorch_ref": [
    "speculative_verify",
    "speculative_verify_pytorch_ref"
  ],
  "alibi_attention": [
    "alibi_attention",
    "alibi_attention"
  ],
  "alibi_attention_pytorch_ref": [
    "alibi_attention",
    "alibi_attention_pytorch_ref"
  ],
  "flash_gqa": [
    "flash_gqa",
    "flash_gqa"
  ],
  "flash_gqa_pytorch_ref": [
    "flash_gqa",
    "flash_gqa_pytorch_ref"
  ],
  "moe_dispatch_gather": [
    "moe_routing",
    "moe_dispatch_gather"
  ],
  "moe_route_pytorch_ref": [
    "moe_routing",
    "moe_route_pytorch_ref"
  ],
  "moe_routing": [
    "moe_routing",
    "moe_routing"
  ],
  "moe_routing_ref": [
    "moe_routing",
    "moe_routing_ref"
  ],
  "moe_routing_pytorch_ref": [
    "moe_routing",
    "moe_routing_pytorch_ref"
  ],
  "moe_dispatch": [
    "moe_routing",
    "moe_dispatch"
  ],
  "moe_gather": [
    "moe_routing",
    "moe_gather"
  ],
  "paged_attention": [
    "paged_attention",
    "paged_attention"
  ],
  "paged_attention_pytorch_reference": [
    "paged_attention",
    "paged_attention_pytorch_reference"
  ],
  "copy_blocks_cow": [
    "paged_kv_cache",
    "copy_blocks_cow"
  ],
  "copy_blocks_cow_pytorch_ref": [
    "paged_kv_cache",
    "copy_blocks_cow_pytorch_ref"
  ],
  "w4a16_gemm": [
    "quantized_gemm",
    "w4a16_gemm"
  ],
  "w4a16_gemm_pytorch_ref": [
    "quantized_gemm",
    "w4a16_gemm_pytorch_ref"
  ],
  "fp8_gemm": [
    "quantized_gemm",
    "fp8_gemm"
  ],
  "fp8_gemm_pytorch_ref": [
    "quantized_gemm",
    "fp8_gemm_pytorch_ref"
  ],
  "selective_scan": [
    "selective_scan",
    "selective_scan"
  ],
  "selective_scan_pytorch_ref": [
    "selective_scan",
    "selective_scan_pytorch_ref"
  ],
  "sliding_window_attn": [
    "sliding_window",
    "sliding_window_attn"
  ],
  "sliding_window_pytorch_ref": [
    "sliding_window",
    "sliding_window_pytorch_ref"
  ],
  "block_sparse_attn": [
    "sparse_attention",
    "block_sparse_attn"
  ],
  "block_sparse_attn_pytorch_ref": [
    "sparse_attention",
    "block_sparse_attn_pytorch_ref"
  ],
  "entmax_bisect": [
    "entmax_bisect",
    "entmax_bisect"
  ],
  "fused_entmax_bisect_pytorch": [
    "entmax_bisect",
    "fused_entmax_bisect_pytorch"
  ],
  "split_head_norm": [
    "split_head_norm",
    "split_head_norm"
  ],
  "split_head_norm_pytorch_ref": [
    "split_head_norm",
    "split_head_norm_pytorch_ref"
  ],
  "aegis_gate_norm": [
    "aegis_gate_norm",
    "aegis_gate_norm"
  ],
  "aegis_gate_norm_pytorch_ref": [
    "aegis_gate_norm",
    "aegis_gate_norm_pytorch_ref"
  ],
  "entropy_pruning": [
    "entropy_pruning",
    "entropy_pruning"
  ],
  "entropy_pruning_pytorch_ref": [
    "entropy_pruning",
    "entropy_pruning_pytorch_ref"
  ],
  "favor_plus": [
    "favor_plus",
    "favor_plus"
  ],
  "favor_plus_pytorch_ref": [
    "favor_plus",
    "favor_plus_pytorch_ref"
  ],
  "fwht_transform": [
    "fwht_transform",
    "fwht_transform"
  ],
  "fast_hadamard_transform": [
    "fwht_transform",
    "fast_hadamard_transform"
  ],
  "rms_variance_match": [
    "rms_variance_match",
    "rms_variance_match"
  ],
  "fused_rms_variance_match_triton": [
    "rms_variance_match",
    "fused_rms_variance_match_triton"
  ],
  "spectral_shrinkage": [
    "spectral_shrinkage",
    "spectral_shrinkage"
  ],
  "spectral_shrinkage_pytorch_ref": [
    "spectral_shrinkage",
    "spectral_shrinkage_pytorch_ref"
  ],
  "adaptive_span": [
    "adaptive_span",
    "adaptive_span"
  ],
  "fused_adaptive_span_penalty_pytorch": [
    "adaptive_span",
    "fused_adaptive_span_penalty_pytorch"
  ],
  "dynamic_conv1d": [
    "dynamic_conv1d",
    "dynamic_conv1d"
  ],
  "fused_dynamic_conv1d_pytorch": [
    "dynamic_conv1d",
    "fused_dynamic_conv1d_pytorch"
  ],
  "paged_recurrent_step": [
    "paged_recurrent_step",
    "paged_recurrent_step"
  ],
  "paged_recurrent_state_step_pytorch": [
    "paged_recurrent_step",
    "paged_recurrent_state_step_pytorch"
  ],
  "paged_kv_evict": [
    "paged_kv_evict",
    "paged_kv_evict"
  ],
  "paged_kv_compact_pytorch_ref": [
    "paged_kv_evict",
    "paged_kv_compact_pytorch_ref"
  ],
  "unified_rope_offset": [
    "unified_rope_offset",
    "unified_rope_offset"
  ],
  "fused_unified_rope_offset_pytorch": [
    "unified_rope_offset",
    "fused_unified_rope_offset_pytorch"
  ],
  "flash_decode_stage2": [
    "flash_decode_stage2",
    "flash_decode_stage2"
  ],
  "flash_decode_stage2_pytorch_ref": [
    "flash_decode_stage2",
    "flash_decode_stage2_pytorch_ref"
  ],
  "svd_prefix_scan": [
    "svd_prefix_scan",
    "svd_prefix_scan"
  ],
  "svd_prefix_scan_pytorch_ref": [
    "svd_prefix_scan",
    "svd_prefix_scan_pytorch_ref"
  ],
  "gdn_delta_rule": [
    "gdn_delta_rule",
    "gdn_delta_rule"
  ],
  "gdn_delta_rule_pytorch_ref": [
    "gdn_delta_rule",
    "gdn_delta_rule_pytorch_ref"
  ],
  "hgrn2_spatial_scan": [
    "hgrn2_spatial_scan",
    "hgrn2_spatial_scan"
  ],
  "hgrn2_spatial_scan_pytorch_ref": [
    "hgrn2_spatial_scan",
    "hgrn2_spatial_scan_pytorch_ref"
  ],
  "gla_unnormalized_scan": [
    "gla_unnormalized_scan",
    "gla_unnormalized_scan"
  ],
  "gla_unnormalized_scan_pytorch_ref": [
    "gla_unnormalized_scan",
    "gla_unnormalized_scan_pytorch_ref"
  ],
  "bitonic_topk": [
    "bitonic_topk",
    "bitonic_topk"
  ],
  "bitonic_topk_pytorch_ref": [
    "bitonic_topk",
    "bitonic_topk_pytorch_ref"
  ],
  "logcumsumexp": [
    "logcumsumexp",
    "logcumsumexp"
  ],
  "logcumsumexp_pytorch_ref": [
    "logcumsumexp",
    "logcumsumexp_pytorch_ref"
  ],
  "soa_attn": [
    "soa_attn",
    "soa_attn"
  ],
  "soa_attn_ref": [
    "soa_attn",
    "soa_attn_ref"
  ],
  "nasc_attention": [
    "nasc_attention",
    "nasc_attention"
  ],
  "nasc_attention_ref": [
    "nasc_attention",
    "nasc_attention_ref"
  ],
  "tpa_attention": [
    "tpa_attention",
    "tpa_attention"
  ],
  "tpa_attention_ref": [
    "tpa_attention",
    "tpa_attention_ref"
  ],
  "spectra_v2": [
    "spectra_v2",
    "spectra_v2"
  ],
  "spectra_v2_ref": [
    "spectra_v2",
    "spectra_v2_ref"
  ],
  "spectraformer": [
    "spectraformer_attention",
    "spectraformer"
  ],
  "spectraformer_ref": [
    "spectraformer_attention",
    "spectraformer_ref"
  ],
  "galerkin_attention": [
    "galerkin_attention",
    "galerkin_attention"
  ],
  "galerkin_attention_ref": [
    "galerkin_attention",
    "galerkin_attention_ref"
  ],
  "hydra_attention": [
    "hydra_attention",
    "hydra_attention"
  ],
  "hydra_attention_ref": [
    "hydra_attention",
    "hydra_attention_ref"
  ],
  "rfa_attention": [
    "rfa_attention",
    "rfa_attention"
  ],
  "rfa_attention_ref": [
    "rfa_attention",
    "rfa_attention_ref"
  ],
  "fast_attention": [
    "fast_attention",
    "fast_attention"
  ],
  "fast_attention_ref": [
    "fast_attention",
    "fast_attention_ref"
  ],
  "linear_rnn_attention": [
    "linear_rnn_attention",
    "linear_rnn_attention"
  ],
  "linear_rnn_attention_ref": [
    "linear_rnn_attention",
    "linear_rnn_attention_ref"
  ],
  "differential_attention": [
    "differential_attention",
    "differential_attention"
  ],
  "differential_attention_ref": [
    "differential_attention",
    "differential_attention_ref"
  ],
  "decay_linear_attention": [
    "decay_linear_attention",
    "decay_linear_attention"
  ],
  "decay_linear_attention_ref": [
    "decay_linear_attention",
    "decay_linear_attention_ref"
  ],
  "hstu_attention": [
    "hstu_attention",
    "hstu_attention"
  ],
  "hstu_attention_ref": [
    "hstu_attention",
    "hstu_attention_ref"
  ],
  "longformer": [
    "longformer",
    "longformer"
  ],
  "longformer_ref": [
    "longformer",
    "longformer_ref"
  ],
  "bigbird": [
    "bigbird",
    "bigbird"
  ],
  "bigbird_ref": [
    "bigbird",
    "bigbird_ref"
  ],
  "reformer": [
    "reformer",
    "reformer"
  ],
  "reformer_ref": [
    "reformer",
    "reformer_ref"
  ],
  "local_attention": [
    "local_attention",
    "local_attention"
  ],
  "local_attention_ref": [
    "local_attention",
    "local_attention_ref"
  ],
  "cluster_attention": [
    "cluster_attention",
    "cluster_attention"
  ],
  "cluster_attention_ref": [
    "cluster_attention",
    "cluster_attention_ref"
  ],
  "routing_attention": [
    "routing_attention",
    "routing_attention"
  ],
  "routing_attention_ref": [
    "routing_attention",
    "routing_attention_ref"
  ],
  "vspec": [
    "vspec",
    "vspec"
  ],
  "vspec_ref": [
    "vspec",
    "vspec_ref"
  ],
  "goldfinch_ahn": [
    "goldfinch_ahn",
    "goldfinch_ahn"
  ],
  "goldfinch_ahn_ref": [
    "goldfinch_ahn",
    "goldfinch_ahn_ref"
  ],
  "hla_attention": [
    "hla_attention",
    "hla_attention"
  ],
  "hla_attention_ref": [
    "hla_attention",
    "hla_attention_ref"
  ],
  "phantom": [
    "phantom",
    "phantom"
  ],
  "phantom_ref": [
    "phantom",
    "phantom_ref"
  ],
  "mamba_block": [
    "mamba_block",
    "mamba_block"
  ],
  "mamba_block_ref": [
    "mamba_block",
    "mamba_block_ref"
  ],
  "selective_ssm": [
    "selective_ssm",
    "selective_ssm"
  ],
  "selective_ssm_ref": [
    "selective_ssm",
    "selective_ssm_ref"
  ],
  "upi_stuffed_mamba": [
    "upi_stuffed_mamba",
    "upi_stuffed_mamba"
  ],
  "upi_stuffed_mamba_ref": [
    "upi_stuffed_mamba",
    "upi_stuffed_mamba_ref"
  ],
  "jamba": [
    "jamba",
    "jamba"
  ],
  "jamba_ref": [
    "jamba",
    "jamba_ref"
  ],
  "nexus": [
    "nexus",
    "nexus"
  ],
  "nexus_ref": [
    "nexus",
    "nexus_ref"
  ],
  "hymba_transx": [
    "hymba_transx",
    "hymba_transx"
  ],
  "hymba_transx_ref": [
    "hymba_transx",
    "hymba_transx_ref"
  ],
  "griffin_ssd": [
    "griffin_ssd",
    "griffin_ssd"
  ],
  "griffin_ssd_ref": [
    "griffin_ssd",
    "griffin_ssd_ref"
  ],
  "jamba_info": [
    "jamba_info",
    "jamba_info"
  ],
  "jamba_info_ref": [
    "jamba_info",
    "jamba_info_ref"
  ],
  "samba_moe": [
    "samba_moe",
    "samba_moe"
  ],
  "samba_moe_ref": [
    "samba_moe",
    "samba_moe_ref"
  ],
  "mamba_svd_flash": [
    "mamba_svd_flash",
    "mamba_svd_flash"
  ],
  "mamba_svd_flash_ref": [
    "mamba_svd_flash",
    "mamba_svd_flash_ref"
  ],
  "samba_m3": [
    "samba_m3",
    "samba_m3"
  ],
  "samba_m3_ref": [
    "samba_m3",
    "samba_m3_ref"
  ],
  "orion": [
    "orion",
    "orion"
  ],
  "orion_ref": [
    "orion",
    "orion_ref"
  ],
  "spectral_mamba": [
    "spectral_mamba",
    "spectral_mamba"
  ],
  "spectral_mamba_ref": [
    "spectral_mamba",
    "spectral_mamba_ref"
  ],
  "simple_ssm": [
    "simple_ssm",
    "simple_ssm"
  ],
  "simple_ssm_ref": [
    "simple_ssm",
    "simple_ssm_ref"
  ],
  "nystrom_attention": [
    "nystrom_attention",
    "nystrom_attention"
  ],
  "nystrom_attention_ref": [
    "nystrom_attention",
    "nystrom_attention_ref"
  ],
  "linformer_attention": [
    "linformer_attention",
    "linformer_attention"
  ],
  "linformer_attention_ref": [
    "linformer_attention",
    "linformer_attention_ref"
  ],
  "rwkv6": [
    "rwkv6",
    "rwkv6"
  ],
  "rwkv6_ref": [
    "rwkv6",
    "rwkv6_ref"
  ],
  "multi_scale_retention": [
    "multi_scale_retention",
    "multi_scale_retention"
  ],
  "multi_scale_retention_pytorch_ref": [
    "multi_scale_retention",
    "multi_scale_retention_pytorch_ref"
  ],
  "liquid_attention": [
    "liquid_attention",
    "liquid_attention"
  ],
  "liquid_attention_ref": [
    "liquid_attention",
    "liquid_attention_ref"
  ],
  "mamba2_block": [
    "mamba2_block",
    "mamba2_block"
  ],
  "mamba2_block_ref": [
    "mamba2_block",
    "mamba2_block_ref"
  ],
  "retnet_retention": [
    "multi_scale_retention",
    "multi_scale_retention"
  ],
  "retnet_retention_pytorch_ref": [
    "multi_scale_retention",
    "multi_scale_retention_pytorch_ref"
  ]
}


CUDA_KERNEL_CATALOG: list[str] = [
    "act_scale_gemm",
    "adaptive_span",
    "aegis_gate_norm",
    "alibi_attention",
    "bigbird",
    "bitonic_topk",
    "block_diag_gemm",
    "block_sparse_attn",
    "causal_conv1d",
    "channel_shuffle",
    "chunked_delta_gate_scan",
    "cluster_attention",
    "cosformer",
    "cosformer_attn",
    "cross_attention",
    "decay_linear_attention",
    "differential_attention",
    "dynamic_conv1d",
    "entmax_bisect",
    "entropy_gate",
    "entropy_pruning",
    "expert_dispatch",
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
    "fused_lowrank_gated_gemm",
    "fused_norm_linear",
    "fused_rope",
    "fwht_transform",
    "galerkin_attention",
    "gdn_delta_rule",
    "gla_scan",
    "gla_unnormalized_scan",
    "goldfinch_ahn",
    "hla_attention",
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
    "kv_cache_copy",
    "latent_kv_attn",
    "layernorm",
    "linear_attn",
    "linear_rnn_attention",
    "linformer_attention",
    "liquid_attention",
    "local_attention",
    "logcumsumexp",
    "longformer",
    "lut_gemm",
    "mamba2_block",
    "mamba_block",
    "mamba_svd_flash",
    "moe_router",
    "multi_scale_retention",
    "nasc_attention",
    "nexus",
    "nystrom_attention",
    "orion",
    "paged_attention",
    "paged_kv_evict",
    "paged_recurrent_step",
    "phantom",
    "quantized_fp8_gemm",
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
    "w4a16_gemm",
]
