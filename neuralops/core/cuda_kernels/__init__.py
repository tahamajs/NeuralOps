"""
NeuralOps Core CUDA Kernels Hardware Operators.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.core.cuda_kernels._bindings import load_cuda_module, get_cuda_arch_flags
from neuralops.core.cuda_kernels._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
)
from neuralops.core.cuda_kernels._utils import (
    is_cuda_available,
    get_device_capability,
    check_cuda_error,
)

# 48 Native CUDA C++ Packages
from neuralops.core.cuda_kernels.low_rank_svd import (
    fused_lowrank_gated_gemm,
    fused_lowrank_gated_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.gdn_scan import (
    chunked_delta_gate_scan,
    chunked_delta_gate_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.hgrn2_scan import (
    hgrn2_fused_bwd_scan,
    hgrn2_fused_bwd_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.cosformer import (
    cosformer_linear_attention,
    cosformer_linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_norm_linear import (
    fused_norm_linear,
    fused_norm_linear_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_rope import (
    fused_rotary_emb,
    fused_rotary_emb_pytorch_ref,
)
from neuralops.core.cuda_kernels.kronecker_gemm import (
    kronecker_matmul,
    kronecker_matmul_pytorch_ref,
)
from neuralops.core.cuda_kernels.lut_gemm import (
    lut_codebook_dequant_gemm,
    lut_codebook_dequant_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.linear_attention import (
    linear_attention,
    linear_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_ce import (
    fused_cross_entropy,
    fused_cross_entropy_pytorch_ref,
)
from neuralops.core.cuda_kernels.rpca_sparse_svd import (
    rpca_sparse_svd_gemm,
    rpca_sparse_svd_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.tt_contraction import (
    tt_contraction,
    tt_contraction_pytorch_ref,
    tt_linear,
    tt_linear_pytorch_ref,
)
from neuralops.core.cuda_kernels.hadamard_gemm import (
    hadamard_w4a16_gemm,
    hadamard_w4a16_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.act_scale_gemm import (
    act_scale_quant_gemm,
    act_scale_quant_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.latent_kv_attn import (
    latent_kv_cache_attention,
    latent_kv_cache_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.entropy_gate import (
    spectral_entropy_svd_gate,
    spectral_entropy_svd_gate_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_kd import (
    fused_kd_loss,
    fused_kd_loss_pytorch_ref,
)
from neuralops.core.cuda_kernels.state_update import (
    recurrent_linear_step,
    recurrent_linear_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_decoding import (
    flash_decoding_attention,
    flash_decoding_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.fms_scan import (
    fms_scan,
    fms_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.rmsnorm import (
    rmsnorm,
    rmsnorm_pytorch_ref,
)
from neuralops.core.cuda_kernels.layernorm import (
    layernorm,
    layernorm_pytorch_ref,
)
from neuralops.core.cuda_kernels.swiglu import (
    swiglu,
    swiglu_pytorch_ref,
)
from neuralops.core.cuda_kernels.fast_gelu import (
    fast_gelu,
    fast_gelu_pytorch_ref,
)
from neuralops.core.cuda_kernels.softplus import (
    fused_softplus,
    fused_softplus_pytorch_ref,
)
from neuralops.core.cuda_kernels.gla_scan import (
    gla_scan,
    gla_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.causal_conv1d import (
    causal_conv1d,
    causal_conv1d_pytorch_ref,
)
from neuralops.core.cuda_kernels.recurrent_gla_step import (
    recurrent_gla_step,
    recurrent_gla_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.recurrent_hgrn2_step import (
    recurrent_hgrn2_step,
    recurrent_hgrn2_step_pytorch_ref,
)
from neuralops.core.cuda_kernels.fused_dropout_add import (
    fused_dropout_add,
    fused_dropout_add_pytorch_ref,
)
from neuralops.core.cuda_kernels.varlen_attention import (
    varlen_attention,
    varlen_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.cross_attention import (
    cross_attention,
    cross_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.rotational_svd import (
    rotational_svd,
    rotational_svd_pytorch_ref,
)
from neuralops.core.cuda_kernels.block_diag_gemm import (
    block_diag_matmul,
    block_diag_matmul_pytorch_ref,
)
from neuralops.core.cuda_kernels.ternary_gemm import (
    ternary_gemm,
    ternary_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.fp4_gemm import (
    fp4_gemm,
    fp4_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.channel_shuffle import (
    channel_shuffle,
    channel_shuffle_pytorch_ref,
)
from neuralops.core.cuda_kernels.speculative_verify import (
    speculative_verify,
    speculative_verify_pytorch_ref,
)
from neuralops.core.cuda_kernels.alibi_attention import (
    alibi_attention,
    alibi_attention_pytorch_ref,
)
from neuralops.core.cuda_kernels.flash_gqa import (
    flash_gqa,
    flash_gqa_pytorch_ref,
)
from neuralops.core.cuda_kernels.moe_routing import (
    moe_dispatch_gather,
    moe_route_pytorch_ref,
    moe_routing,
    moe_routing_ref,
    moe_routing_pytorch_ref,
    moe_dispatch,
    moe_gather,
)
from neuralops.core.cuda_kernels.paged_attention import (
    paged_attention,
    paged_attention_pytorch_reference,
)
from neuralops.core.cuda_kernels.paged_kv_cache import (
    copy_blocks_cow,
    copy_blocks_cow_pytorch_ref,
)
from neuralops.core.cuda_kernels.quantized_gemm import (
    w4a16_gemm,
    w4a16_gemm_pytorch_ref,
    fp8_gemm,
    fp8_gemm_pytorch_ref,
)
from neuralops.core.cuda_kernels.selective_scan import (
    selective_scan,
    selective_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.sliding_window import (
    sliding_window_attn,
    sliding_window_pytorch_ref,
)
from neuralops.core.cuda_kernels.sparse_attention import (
    block_sparse_attn,
    block_sparse_attn_pytorch_ref,
)

from neuralops.core.cuda_kernels.entmax_bisect import (
    entmax_bisect,
    fused_entmax_bisect_pytorch,
)
from neuralops.core.cuda_kernels.split_head_norm import (
    split_head_norm,
    split_head_norm_pytorch_ref,
)
from neuralops.core.cuda_kernels.aegis_gate_norm import (
    aegis_gate_norm,
    aegis_gate_norm_pytorch_ref,
)
from neuralops.core.cuda_kernels.entropy_pruning import (
    entropy_pruning,
    entropy_pruning_pytorch_ref,
)
from neuralops.core.cuda_kernels.favor_plus import (
    favor_plus,
    favor_plus_pytorch_ref,
)
from neuralops.core.cuda_kernels.fwht_transform import (
    fwht_transform,
    fast_hadamard_transform,
)
from neuralops.core.cuda_kernels.rms_variance_match import (
    rms_variance_match,
    fused_rms_variance_match_triton,
)
from neuralops.core.cuda_kernels.spectral_shrinkage import (
    spectral_shrinkage,
    spectral_shrinkage_pytorch_ref,
)
from neuralops.core.cuda_kernels.adaptive_span import (
    adaptive_span,
    fused_adaptive_span_penalty_pytorch,
)
from neuralops.core.cuda_kernels.dynamic_conv1d import (
    dynamic_conv1d,
    fused_dynamic_conv1d_pytorch,
)
from neuralops.core.cuda_kernels.paged_recurrent_step import (
    paged_recurrent_step,
    paged_recurrent_state_step_pytorch,
)
from neuralops.core.cuda_kernels.paged_kv_evict import (
    paged_kv_evict,
    paged_kv_compact_pytorch_ref,
)
from neuralops.core.cuda_kernels.unified_rope_offset import (
    unified_rope_offset,
    fused_unified_rope_offset_pytorch,
)
from neuralops.core.cuda_kernels.flash_decode_stage2 import (
    flash_decode_stage2,
    flash_decode_stage2_pytorch_ref,
)
from neuralops.core.cuda_kernels.svd_prefix_scan import (
    svd_prefix_scan,
    svd_prefix_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.gdn_delta_rule import (
    gdn_delta_rule,
    gdn_delta_rule_pytorch_ref,
)
from neuralops.core.cuda_kernels.hgrn2_spatial_scan import (
    hgrn2_spatial_scan,
    hgrn2_spatial_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.gla_unnormalized_scan import (
    gla_unnormalized_scan,
    gla_unnormalized_scan_pytorch_ref,
)
from neuralops.core.cuda_kernels.bitonic_topk import (
    bitonic_topk,
    bitonic_topk_pytorch_ref,
)
from neuralops.core.cuda_kernels.logcumsumexp import (
    logcumsumexp,
    logcumsumexp_pytorch_ref,
)

# 20 Specialized Attention Modules CUDA Kernels

from neuralops.core.cuda_kernels.soa_attn import (
    soa_attn,
    soa_attn_ref,
)
from neuralops.core.cuda_kernels.nasc_attention import (
    nasc_attention,
    nasc_attention_ref,
)
from neuralops.core.cuda_kernels.tpa_attention import (
    tpa_attention,
    tpa_attention_ref,
)
from neuralops.core.cuda_kernels.spectra_v2 import (
    spectra_v2,
    spectra_v2_ref,
)
from neuralops.core.cuda_kernels.spectraformer_attention import (
    spectraformer,
    spectraformer_ref,
)
from neuralops.core.cuda_kernels.galerkin_attention import (
    galerkin_attention,
    galerkin_attention_ref,
)
from neuralops.core.cuda_kernels.hydra_attention import (
    hydra_attention,
    hydra_attention_ref,
)
from neuralops.core.cuda_kernels.rfa_attention import (
    rfa_attention,
    rfa_attention_ref,
)
from neuralops.core.cuda_kernels.fast_attention import (
    fast_attention,
    fast_attention_ref,
)
from neuralops.core.cuda_kernels.linear_rnn_attention import (
    linear_rnn_attention,
    linear_rnn_attention_ref,
)
from neuralops.core.cuda_kernels.differential_attention import (
    differential_attention,
    differential_attention_ref,
)
from neuralops.core.cuda_kernels.decay_linear_attention import (
    decay_linear_attention,
    decay_linear_attention_ref,
)
from neuralops.core.cuda_kernels.hstu_attention import (
    hstu_attention,
    hstu_attention_ref,
)
from neuralops.core.cuda_kernels.longformer import (
    longformer,
    longformer_ref,
)
from neuralops.core.cuda_kernels.bigbird import (
    bigbird,
    bigbird_ref,
)
from neuralops.core.cuda_kernels.reformer import (
    reformer,
    reformer_ref,
)
from neuralops.core.cuda_kernels.local_attention import (
    local_attention,
    local_attention_ref,
)
from neuralops.core.cuda_kernels.cluster_attention import (
    cluster_attention,
    cluster_attention_ref,
)
from neuralops.core.cuda_kernels.routing_attention import (
    routing_attention,
    routing_attention_ref,
)

# 20 Additional SSM, Recurrent & Global Attention CUDA Kernels
from neuralops.core.cuda_kernels.vspec import (
    vspec,
    vspec_ref,
)
from neuralops.core.cuda_kernels.goldfinch_ahn import (
    goldfinch_ahn,
    goldfinch_ahn_ref,
)
from neuralops.core.cuda_kernels.phantom import (
    phantom,
    phantom_ref,
)
from neuralops.core.cuda_kernels.mamba_block import (
    mamba_block,
    mamba_block_ref,
)
from neuralops.core.cuda_kernels.selective_ssm import (
    selective_ssm,
    selective_ssm_ref,
)
from neuralops.core.cuda_kernels.upi_stuffed_mamba import (
    upi_stuffed_mamba,
    upi_stuffed_mamba_ref,
)
from neuralops.core.cuda_kernels.jamba import (
    jamba,
    jamba_ref,
)
from neuralops.core.cuda_kernels.nexus import (
    nexus,
    nexus_ref,
)
from neuralops.core.cuda_kernels.hymba_transx import (
    hymba_transx,
    hymba_transx_ref,
)
from neuralops.core.cuda_kernels.griffin_ssd import (
    griffin_ssd,
    griffin_ssd_ref,
)
from neuralops.core.cuda_kernels.jamba_info import (
    jamba_info,
    jamba_info_ref,
)
from neuralops.core.cuda_kernels.samba_moe import (
    samba_moe,
    samba_moe_ref,
)
from neuralops.core.cuda_kernels.mamba_svd_flash import (
    mamba_svd_flash,
    mamba_svd_flash_ref,
)
from neuralops.core.cuda_kernels.samba_m3 import (
    samba_m3,
    samba_m3_ref,
)
from neuralops.core.cuda_kernels.orion import (
    orion,
    orion_ref,
)
from neuralops.core.cuda_kernels.spectral_mamba import (
    spectral_mamba,
    spectral_mamba_ref,
)
from neuralops.core.cuda_kernels.simple_ssm import (
    simple_ssm,
    simple_ssm_ref,
)
from neuralops.core.cuda_kernels.nystrom_attention import (
    nystrom_attention,
    nystrom_attention_ref,
)
from neuralops.core.cuda_kernels.linformer_attention import (
    linformer_attention,
    linformer_attention_ref,
)
from neuralops.core.cuda_kernels.rwkv6 import (
    rwkv6,
    rwkv6_ref,
)

from neuralops.core.cuda_kernels.multi_scale_retention import (
    multi_scale_retention,
    multi_scale_retention_pytorch_ref,
)
retnet_retention = multi_scale_retention
retnet_retention_pytorch_ref = multi_scale_retention_pytorch_ref

from neuralops.core.cuda_kernels.liquid_attention import (
    liquid_attention,
    liquid_attention_ref,
)
from neuralops.core.cuda_kernels.mamba2_block import (
    mamba2_block,
    mamba2_block_ref,
)

def hgrn2_gated_deltanet(*args, **kwargs):
    from neuralops.attentions.gated.hgrn2_gated_deltanet import HGRN2GatedDeltaNet
    return HGRN2GatedDeltaNet(*args, **kwargs)

def gated_deltanet_block(*args, **kwargs):
    from neuralops.attentions.gated.gated_deltanet_block import GatedDeltaNetBlock
    return GatedDeltaNetBlock(*args, **kwargs)

def hgrn2_layer(*args, **kwargs):
    from neuralops.attentions.gated.hgrn2_gated_layer import HGRN2Layer
    return HGRN2Layer(*args, **kwargs)

def hgrn2_gated_outer(*args, **kwargs):
    from neuralops.attentions.gated.hgrn2_gated_outer import HGRN2GatedOuterBlock
    return HGRN2GatedOuterBlock(*args, **kwargs)


# Canonical Complete Inventory of 114 Native CUDA C++ Kernels (.cu files)
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
        import neuralops.core.cuda_kernels as nck
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


__all__ = [
    "load_cuda_module",
    "get_cuda_arch_flags",
    "dispatch",
    "force_kernel_backend",
    "get_dispatch_stats",
    "reset_dispatch_stats",
    "is_cuda_available",
    "get_device_capability",
    "check_cuda_error",
    "CUDA_KERNEL_CATALOG",
    "list_cuda_kernels",
    # 48 Flagship CUDA C++ Operators
    "fused_lowrank_gated_gemm",
    "fused_lowrank_gated_gemm_pytorch_ref",
    "chunked_delta_gate_scan",
    "chunked_delta_gate_scan_pytorch_ref",
    "hgrn2_fused_bwd_scan",
    "hgrn2_fused_bwd_scan_pytorch_ref",
    "cosformer_linear_attention",
    "cosformer_linear_attention_pytorch_ref",
    "fused_norm_linear",
    "fused_norm_linear_pytorch_ref",
    "fused_rotary_emb",
    "fused_rotary_emb_pytorch_ref",
    "kronecker_matmul",
    "kronecker_matmul_pytorch_ref",
    "lut_codebook_dequant_gemm",
    "lut_codebook_dequant_gemm_pytorch_ref",
    "linear_attention",
    "linear_attention_pytorch_ref",
    "fused_cross_entropy",
    "fused_cross_entropy_pytorch_ref",
    "rpca_sparse_svd_gemm",
    "rpca_sparse_svd_gemm_pytorch_ref",
    "tt_contraction",
    "tt_contraction_pytorch_ref",
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
    "fms_scan",
    "fms_scan_pytorch_ref",
    "rmsnorm",
    "rmsnorm_pytorch_ref",
    "layernorm",
    "layernorm_pytorch_ref",
    "swiglu",
    "swiglu_pytorch_ref",
    "fast_gelu",
    "fast_gelu_pytorch_ref",
    "fused_softplus",
    "fused_softplus_pytorch_ref",
    "gla_scan",
    "gla_scan_pytorch_ref",
    "causal_conv1d",
    "causal_conv1d_pytorch_ref",
    "recurrent_gla_step",
    "recurrent_gla_step_pytorch_ref",
    "recurrent_hgrn2_step",
    "recurrent_hgrn2_step_pytorch_ref",
    "fused_dropout_add",
    "fused_dropout_add_pytorch_ref",
    "varlen_attention",
    "varlen_attention_pytorch_ref",
    "cross_attention",
    "cross_attention_pytorch_ref",
    "rotational_svd",
    "rotational_svd_pytorch_ref",
    "block_diag_matmul",
    "block_diag_matmul_pytorch_ref",
    "ternary_gemm",
    "ternary_gemm_pytorch_ref",
    "fp4_gemm",
    "fp4_gemm_pytorch_ref",
    "channel_shuffle",
    "channel_shuffle_pytorch_ref",
    "speculative_verify",
    "speculative_verify_pytorch_ref",
    "alibi_attention",
    "alibi_attention_pytorch_ref",
    # Additional Native CUDA Operators
    "flash_gqa",
    "flash_gqa_pytorch_ref",
    "moe_dispatch_gather",
    "moe_route_pytorch_ref",
    "paged_attention",
    "paged_attention_pytorch_reference",
    "copy_blocks_cow",
    "copy_blocks_cow_pytorch_ref",
    "w4a16_gemm",
    "w4a16_gemm_pytorch_ref",
    "fp8_gemm",
    "fp8_gemm_pytorch_ref",
    "selective_scan",
    "selective_scan_pytorch_ref",
    "sliding_window_attn",
    "sliding_window_pytorch_ref",
    "block_sparse_attn",
    "block_sparse_attn_pytorch_ref",
    # 20 Additional Ultra-Optimized CUDA Kernels
    "entmax_bisect",
    "fused_entmax_bisect_pytorch",
    "split_head_norm",
    "split_head_norm_pytorch_ref",
    "aegis_gate_norm",
    "aegis_gate_norm_pytorch_ref",
    "entropy_pruning",
    "entropy_pruning_pytorch_ref",
    "favor_plus",
    "favor_plus_pytorch_ref",
    "fwht_transform",
    "fast_hadamard_transform",
    "rms_variance_match",
    "fused_rms_variance_match_triton",
    "spectral_shrinkage",
    "spectral_shrinkage_pytorch_ref",
    "adaptive_span",
    "fused_adaptive_span_penalty_pytorch",
    "dynamic_conv1d",
    "fused_dynamic_conv1d_pytorch",
    "paged_recurrent_step",
    "paged_recurrent_state_step_pytorch",
    "paged_kv_evict",
    "paged_kv_compact_pytorch_ref",
    "unified_rope_offset",
    "fused_unified_rope_offset_pytorch",
    "flash_decode_stage2",
    "flash_decode_stage2_pytorch_ref",
    "svd_prefix_scan",
    "svd_prefix_scan_pytorch_ref",
    "gdn_delta_rule",
    "gdn_delta_rule_pytorch_ref",
    "hgrn2_spatial_scan",
    "hgrn2_spatial_scan_pytorch_ref",
    "gla_unnormalized_scan",
    "gla_unnormalized_scan_pytorch_ref",
    "bitonic_topk",
    "bitonic_topk_pytorch_ref",
    "logcumsumexp",
    "logcumsumexp_pytorch_ref",
    # 20 Specialized Attention Modules CUDA Kernels
    "soa_attn",
    "soa_attn_ref",
    "nasc_attention",
    "nasc_attention_ref",
    "tpa_attention",
    "tpa_attention_ref",
    "spectra_v2",
    "spectra_v2_ref",
    "spectraformer",
    "spectraformer_ref",
    "galerkin_attention",
    "galerkin_attention_ref",
    "hydra_attention",
    "hydra_attention_ref",
    "rfa_attention",
    "rfa_attention_ref",
    "fast_attention",
    "fast_attention_ref",
    "linear_rnn_attention",
    "linear_rnn_attention_ref",
    "differential_attention",
    "differential_attention_ref",
    "decay_linear_attention",
    "decay_linear_attention_ref",
    "hstu_attention",
    "hstu_attention_ref",
    "longformer",
    "longformer_ref",
    "bigbird",
    "bigbird_ref",
    "reformer",
    "reformer_ref",
    "local_attention",
    "local_attention_ref",
    "cluster_attention",
    "cluster_attention_ref",
    "routing_attention",
    "routing_attention_ref",
    # 20 Additional SSM, Recurrent & Global Attention CUDA Kernels
    "vspec",
    "vspec_ref",
    "goldfinch_ahn",
    "goldfinch_ahn_ref",
    "phantom",
    "phantom_ref",
    "mamba_block",
    "mamba_block_ref",
    "selective_ssm",
    "selective_ssm_ref",
    "upi_stuffed_mamba",
    "upi_stuffed_mamba_ref",
    "jamba",
    "jamba_ref",
    "nexus",
    "nexus_ref",
    "hymba_transx",
    "hymba_transx_ref",
    "griffin_ssd",
    "griffin_ssd_ref",
    "jamba_info",
    "jamba_info_ref",
    "samba_moe",
    "samba_moe_ref",
    "mamba_svd_flash",
    "mamba_svd_flash_ref",
    "samba_m3",
    "samba_m3_ref",
    "orion",
    "orion_ref",
    "spectral_mamba",
    "spectral_mamba_ref",
    "simple_ssm",
    "simple_ssm_ref",
    "nystrom_attention",
    "nystrom_attention_ref",
    "linformer_attention",
    "linformer_attention_ref",
    "rwkv6",
    "rwkv6_ref",
    "multi_scale_retention",
    "multi_scale_retention_pytorch_ref",
    "retnet_retention",
    "retnet_retention_pytorch_ref",
    "liquid_attention",
    "liquid_attention_ref",
    "mamba2_block",
    "mamba2_block_ref",
    "hgrn2_gated_deltanet",
    "gated_deltanet_block",
    "hgrn2_layer",
    "hgrn2_gated_outer",
]




