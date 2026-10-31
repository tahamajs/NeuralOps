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
    "aegis_hybrid_attention": "aegis_hybrid",
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
    "aasvd": "aa_svd_compressor",
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
    "svd_llm": "svd_llm_compressor",
}


def handle_deprecated_alias(name: str) -> str:
    """Resolve deprecated alias recursively with standard deprecation warning."""
    k = name.lower()
    from neuralops.core.registry import DEPRECATED_ALIASES as REG_DEPRECATED
    if k in REG_DEPRECATED:
        new_name = REG_DEPRECATED[k]
        warnings.warn(
            f"'{name}' is deprecated. Use '{new_name}' instead.",
            DeprecationWarning,
            stacklevel=3,
        )
        return new_name
    return name
