#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Legacy Module Import Aliases

Backward-compatibility alias registry mapping deprecated module paths to modern namespaces.
"""

from __future__ import annotations

import json
import os
import warnings

_DEPRECATED_ALIASES_PATH = os.path.join(os.path.dirname(__file__), "deprecated_aliases.json")
if os.path.exists(_DEPRECATED_ALIASES_PATH):
    with open(_DEPRECATED_ALIASES_PATH) as _fp:
        DEPRECATED_ALIASES: dict[str, str] = json.load(_fp)
else:
    DEPRECATED_ALIASES = {}

ATTENTION_FILE_MAP: dict[str, str] = {
    "aegis_v6": "gated_hybrid_kernel_attention",
    "native_hybrid_attention": "nha_attention",
    "svd_seq": "svd_attention",
    "gated_linear_attention": "decay_linear_attention",
    "gated_linear_attn": "decay_linear_attention",
    "spectron_fp8": "spectron_fp8_attention",
    "gqa": "gqa_attention",
    "galerkin": "galerkin_attention",
    "galerkin_inject": "galerkin_attention",
    "adaptive_span": "adaptive_span_attention",
    "adaptive_span_inject": "adaptive_span_attention",
    "adaptive_sparse_inject": "adaptive_sparse_attention",
    "adaptive_svd": "adaptive_svd_attention",
    "asvd": "asvd_attention",
    "adasplash_inject": "adasplash_attention",
    "dynamic_conv": "dynamic_conv_attention",
    "aegis_hybrid": "tri_branch_hybrid_attention",
    "aegis_hybrid_attention": "tri_branch_hybrid_attention",
    "tri_branch_hybrid": "tri_branch_hybrid_attention",
    "cluster_inject": "cluster_attention",
    "compressed_mha": "compressed_mha_attention",
    "cosformer_inject": "cosformer",
    "hla_inject": "hla_attention",
    "kda_inject": "kda_attention",
    "hybrid_local_kernel": "hybrid_local_kernel_attention",
    "split_head_hybrid": "split_head_hybrid_attention",
    "gated_hybrid_kernel": "gated_hybrid_kernel_attention",
    "positive_kernel_linear": "positive_kernel_linear_attention",
    "landmark_local": "landmark_local_attention",
    "refined_landmark": "refined_landmark_attention",
    "mamba_attention": "mamba_block",
    "moba": "moba_attention",
    "nha_compressor": "nha_attention",
    "rpc": "rpc_attention",
    "salt": "salt_attention",
    "soa_fused_module": "SOAFusedAttention",
    "soa_module": "soa_attn",
    "spectra_v2_attention": "spectra_v2",
    "svd_compressed_attention": "svd_compressed",
    "svd_llm_v2": "svd_llm_v2_attention",
    "svd_proj_attn": "svd_projection_attention",
    "linear_attention": "svd_linear_attention",
    "inject_svd_linear_attention": "svd_linear_attention",
    "token_compressed": "token_compressed_attention",
    "tt": "token_compressed_attention",
    "upi_mamba": "upi_stuffed_mamba",
    "gdn2": "gated.flagship_fla",
    "gdn2_attention": "gated.flagship_fla",
    "pgdn": "gated.flagship_fla",
    "pgdn_attention": "gated.flagship_fla",
    "preconditioned_gdn": "gated.flagship_fla",
    "preconditioned_gdn_attention": "gated.flagship_fla",
    "delta_product": "gated.flagship_fla",
    "delta_product_attention": "gated.flagship_fla",
    "deltaproduct": "gated.flagship_fla",
    "path_attention": "gated.flagship_fla",
    "weighted_ensemble": "hybrid.hybrid_attention",
    "smart_weighted_ensemble": "hybrid.hybrid_attention",
}

COMPRESSOR_FILE_MAP: dict[str, str] = {
    "asvd": "asvd_compressor",
    "asvd_compressor": "asvd_compressor",
    "auto_svd": "auto_svd",
    "awq": "awq",
    "awq_compressor": "awq",
    "awq_real": "awq",
    "fp8": "fp8_compressor",
    "fp8_compressor": "fp8_compressor",
    "gptq": "gptq",
    "gptq_compressor": "gptq",
    "gptq_real": "gptq",
    "hadamard_incoherent_svd": "hadamard_incoherent_svd",
    "hadamard_svd": "hadamard_incoherent_svd",
    "lillama": "lillama_compressor",
    "lillama_compressor": "lillama_compressor",
    "quarot": "quarot",
    "quarot_compressor": "quarot",
    "quarot_svd": "hadamard_incoherent_svd",
    "smoothquant": "smoothquant",
    "smoothquant_compressor": "smoothquant",
    "smoothquant_real": "smoothquant",
    "sparsegpt": "sparsegpt",
    "wanda": "wanda",
    "wanda_compressor": "wanda",
    "wanda_pruner": "wanda",
    "aasvd": "aa_svd_compressor",
    "gfws": "gfws_d_compressor",
    "gfws_d": "gfws_d_compressor",
    "mgie": "mgie_svd_compressor",
    "mgie_budget": "mgie_svd_compressor",
    "mgie_svd": "mgie_svd_compressor",
    "adasvd": "adasvd_compressor",
    "cur_moe": "cur_moe_compressor",
    "frandi": "frandi_compressor",
    "grecodec": "grecodec_compressor",
    "hybridkv": "hybridkv_compressor",
    "kvtc": "kvtc_compressor",
    "modegpt": "modegpt_compressor",
    "oats": "oats_compressor",
    "proxsparse_pruner": "proxsparse",
    "rpca": "rpca_compressor",
    "saes_svd": "saes_svd_v2",
    "saes_svd_compressor": "saes_svd_v2",
    "sdrm": "sdrm_compressor",
    "slicegpt_compressor": "slicegpt",
    "soka": "soka_compressor",
    "swift_svd_compressor": "swift_svd_act",
    "zerokv_neo": "zerokv_neo_compressor",
    "dynamic_entropy_svd": "dynamic_rank",
    "entropy_rank_allocator": "dynamic_rank",
    "res_svd": "residual_svd_compressor",
    "spectral_residual_lora": "residual_svd_compressor",
    "svd": "direct_svd",
    "hnc_svd": "direct_svd",
    "hnc_svd_compressor": "direct_svd",
    "svd_llm": "svd_llm_compressor",
}


def handle_deprecated_alias(name: str) -> str:
    """Resolve deprecated alias recursively with standard deprecation warning."""
    k = name.lower()
    if k in DEPRECATED_ALIASES:
        new_name = DEPRECATED_ALIASES[k]
        if not os.environ.get("NEURALOPS_NO_DEPRECATION_WARNINGS"):
            warnings.warn(
                f"'{name}' is deprecated and will be removed in v3.0.0; use '{new_name}' instead.",
                DeprecationWarning,
                stacklevel=3,
            )
        return new_name
    return name
