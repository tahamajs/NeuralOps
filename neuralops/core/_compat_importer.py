#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Transparent In-Memory Import Interceptor for NeuralOps Legacy Subpackages.
Ensures identical module identity (A is B) without maintaining duplicate files.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from types import ModuleType
from typing import Any, Optional, Sequence

logger = logging.getLogger("NeuralOps.CompatImporter")

_ATTENTIONS_LEGACY_MAP = {
    "lola_attention": "adaptive.lola_attention",
    "eat": "adaptive.eat",
    "progressive_token_pruning": "adaptive.progressive_token_pruning",
    "infini_attention": "adaptive.infini_attention",
    "span_expanded_attention": "adaptive.span_expanded_attention",
    "adaptive_span_attention": "adaptive.adaptive_span_attention",
    "squeezing_heads_distill": "adaptive.squeezing_heads_distill",
    "token_compressed_attention": "adaptive.token_compressed_attention",
    "block_transformer_hiba": "adaptive.block_transformer_hiba",
    "full_attention": "dense.full_attention",
    "zeros_attention": "dense.zeros_attention",
    "template_attention": "experimental.template_attention",
    "xattention": "fast_kernels.xattention",
    "flash_attention3_fp8": "fast_kernels.flash_attention3_fp8",
    "soa_attn": "fast_kernels.soa_attn",
    "flash_attention3": "fast_kernels.flash_attention3",
    "adasplash_attention": "fast_kernels.adasplash_attention",
    "gated_deltanet_block": "gated.gated_deltanet_block",
    "kda_attention": "gated.kda_attention",
    "hgrn2_gated_outer": "gated.hgrn2_gated_outer",
    "hgrn2_gated_deltanet": "gated.hgrn2_gated_deltanet",
    "hgrn2_gated_layer": "gated.hgrn2_gated_layer",
    "gated_delta_net": "gated.gated_delta_net",
    "hla_attention": "gated.hla_attention",
    "decay_linear_attention": "gated.decay_linear_attention",
    "regla_attention": "hybrid.regla_attention",
    "gated_hybrid_kernel_attention": "hybrid.gated_hybrid_kernel_attention",
    "spectron_fp8_attention": "hybrid.spectron_fp8_attention",
    "optimized_hybrid_attention": "hybrid.optimized_hybrid_attention",
    "mixattention_moska": "hybrid.mixattention_moska",
    "granite_hybrid": "hybrid.granite_hybrid",
    "landmark_local_attention": "hybrid.landmark_local_attention",
    "refined_landmark_attention": "hybrid.refined_landmark_attention",
    "split_head_hybrid_attention": "hybrid.split_head_hybrid_attention",
    "fusion_attn": "hybrid.fusion_attn",
    "tri_branch_hybrid_attention": "hybrid.tri_branch_hybrid_attention",
    "positive_kernel_linear_attention": "hybrid.positive_kernel_linear_attention",
    "halo_regla": "hybrid.halo_regla",
    "hybrid_local_kernel_attention": "hybrid.hybrid_local_kernel_attention",
    "hybrid_attention": "hybrid.hybrid_attention",
    "linformer_attention": "linear.linformer_attention",
    "cosformer": "linear.cosformer",
    "nystrom_attention": "linear.nystrom_attention",
    "nasc_attention": "linear.nasc_attention",
    "tpa_attention": "linear.tpa_attention",
    "spectra_v2": "linear.spectra_v2",
    "spectraformer_attention": "linear.spectraformer_attention",
    "spectron": "linear.spectron",
    "performer_attention": "linear.performer_attention",
    "tlinformer_attention": "linear.tlinformer_attention",
    "primal_attention": "linear.primal_attention",
    "rala_attention": "linear.rala_attention",
    "rpc_attention": "linear.rpc_attention",
    "galerkin_attention": "linear.galerkin_attention",
    "triton_cosformer": "linear.triton_cosformer",
    "chela_slay": "linear.chela_slay",
    "unified_rope": "modern.unified_rope",
    "gqa_attention": "modern.gqa_attention",
    "vig_attention": "modern.vig_attention",
    "mla_attention": "modern.mla_attention",
    "wgqa_attention": "modern.wgqa_attention",
    "qcqa_attention": "modern.qcqa_attention",
    "nha_attention": "modern.nha_attention",
    "dynamic_conv": "modern.dynamic_conv_attention",
    "dynamic_conv_attention": "modern.dynamic_conv_attention",
    "asvd": "svd.asvd_attention",
    "padre_attention": "sparse.padre_attention",
    "moba_attention": "sparse.moba_attention",
    "routing_attention": "sparse.routing_attention",
    "moe_deltakv_paged": "sparse.moe_deltakv_paged",
    "cluster_attention": "sparse.cluster_attention",
    "lsh_attention": "sparse.lsh_attention",
    "reap_moe": "sparse.reap_moe",
    "nsa_attention": "sparse.nsa_attention",
    "bam_umoe": "sparse.bam_umoe",
    "hamming_attention": "sparse.hamming_attention",
    "adaptive_sparse_attention": "sparse.adaptive_sparse_attention",
    "expert_choice_moe": "sparse.expert_choice_moe",
    "vspec": "ssm.vspec",
    "goldfinch_ahn": "ssm.goldfinch_ahn",
    "phantom": "ssm.phantom",
    "mamba_block": "ssm.mamba_block",
    "selective_ssm": "ssm.selective_ssm",
    "upi_stuffed_mamba": "ssm.upi_stuffed_mamba",
    "jamba": "ssm.jamba",
    "nexus": "ssm.nexus",
    "hymba_transx": "ssm.hymba_transx",
    "griffin_ssd": "ssm.griffin_ssd",
    "jamba_info": "ssm.jamba_info",
    "samba_moe": "ssm.samba_moe",
    "mamba_svd_flash": "ssm.mamba_svd_flash",
    "samba_m3": "ssm.samba_m3",
    "orion": "ssm.orion",
    "spectral_mamba": "ssm.spectral_mamba",
    "simple_ssm": "ssm.simple_ssm",
    "compressed_attention": "svd.compressed_attention",
    "svd_projection_attention": "svd.svd_projection_attention",
    "fused_svd_kernel": "svd.fused_svd_kernel",
    "compressed_mha_attention": "svd.compressed_mha_attention",
    "factorised_matrix_state": "svd.factorised_matrix_state",
    "projection_factorized_attn": "svd.projection_factorized_attn",
    "svd_attention": "svd.svd_attention",
    "svd_compressed": "svd.svd_compressed",
    "triton_svd_linear_attention": "svd.triton_svd_linear_attention",
    "asvd_attention": "svd.asvd_attention",
    "naive_svd": "svd.naive_svd",
    "residual_svd_attention": "svd.residual_svd_attention",
    "minimal_svd": "svd.minimal_svd",
    "lora_attention": "svd.lora_attention",
    "svd_linear_decay": "svd.svd_linear_decay",
    "svd_llm_v2_attention": "svd.svd_llm_v2_attention",
    "salt_attention": "svd.salt_attention",
    "svd_mamba": "svd.svd_mamba",
    "minimal_nova": "svd.minimal_nova",
    "svd_linear_attention": "svd.svd_linear_attention",
    "adaptive_svd_attention": "svd.adaptive_svd_attention",
}

_ATTENTIONS_RENAMED_MAP = {
    "aegis_hybrid": (
        "hybrid.tri_branch_hybrid_attention",
        {"AegisHybridAttention": "TriBranchHybridAttention", "inject_aegis_hybrid": "inject_tri_branch_hybrid"},
    ),
    "aegis_spectron_fp8": (
        "hybrid.spectron_fp8_attention",
        {"AegisSpectronFP8Attention": "SpectronFP8Attention", "inject_aegis_spectron_fp8": "inject_spectron_fp8"},
    ),
    "aegis_v2": (
        "hybrid.hybrid_local_kernel_attention",
        {"AegisAttentionV2": "HybridLocalKernelAttention", "inject_aegis_v2": "inject_hybrid_local_kernel"},
    ),
    "aegis_v5": (
        "hybrid.split_head_hybrid_attention",
        {
            "AegisAttention": "SplitHeadHybridAttention",
            "AegisConfigV5": "SplitHeadHybridConfig",
            "inject_aegis_v5": "inject_split_head_hybrid",
        },
    ),
    "aegis_v6": (
        "hybrid.gated_hybrid_kernel_attention",
        {"AegisAttentionV3": "GatedHybridKernelAttention", "inject_aegis_v6": "inject_gated_hybrid_kernel"},
    ),
    "leka_v1": (
        "hybrid.positive_kernel_linear_attention",
        {"LEKAV1Attention": "PositiveKernelLinearAttention", "inject_leka_v1": "inject_positive_kernel_linear"},
    ),
    "leka_v2": (
        "hybrid.landmark_local_attention",
        {"LEKAv2Attention": "LandmarkLocalAttention", "inject_leka_v2": "inject_landmark_local"},
    ),
    "leka_v3": (
        "hybrid.refined_landmark_attention",
        {"LEKAv3Attention": "RefinedLandmarkAttention", "inject_leka_v3": "inject_refined_landmark"},
    ),
    "res_svd": (
        "svd.residual_svd_attention",
        {"ResSVDAttention": "ResidualSVDAttention", "inject_res_svd": "inject_residual_svd_attention"},
    ),
}

_COMPRESSION_RENAMED_MAP = {
    "res_svd": (
        "svd.residual_svd_compressor",
        {
            "ResSVDCompressor": "ResidualSVDCompressor",
            "inject_res_svd": "inject_residual_svd_compressor",
            "ResSVDLinear": "ResidualSVDLinear",
        },
    )
}


class ExactModuleAliasLoader(Loader):
    """Loader that delegates module resolution directly to canonical module or builds an alias module."""

    def __init__(self, canonical_name: str, is_pkg: bool = False, symbol_aliases: Optional[dict[str, str]] = None):
        self.canonical_name = canonical_name
        self.is_pkg = is_pkg
        self.symbol_aliases = symbol_aliases

    def create_module(self, spec: Any) -> Optional[ModuleType]:
        if not self.symbol_aliases:
            if self.canonical_name not in sys.modules:
                importlib.import_module(self.canonical_name)
            return sys.modules[self.canonical_name]

        canon_mod = importlib.import_module(self.canonical_name)
        alias_mod = types.ModuleType(spec.name)
        alias_mod.__dict__.update(canon_mod.__dict__)
        for alias_sym, canon_sym in self.symbol_aliases.items():
            if hasattr(canon_mod, canon_sym):
                setattr(alias_mod, alias_sym, getattr(canon_mod, canon_sym))
        return alias_mod

    def exec_module(self, module: ModuleType) -> None:
        pass


class NeuralOpsCompatibilityRedirectFinder(MetaPathFinder):
    """Intercepts legacy subpackage requests and resolves them to canonical packages with identity preservation."""

    _EXACT_MAP = {
        "neuralops.attentions": ("neuralops.layers", True),
        "neuralops.compression_engine": ("neuralops.compression", True),
        "neuralops.training": ("neuralops.trainer", True),
        "neuralops.backward_compat": ("neuralops.core", True),
    }

    _PREFIX_MAP = {
        "neuralops.attentions.": "neuralops.layers.",
        "neuralops.compression_engine.": "neuralops.compression.",
        "neuralops.training.": "neuralops.trainer.",
        "neuralops.backward_compat.attentions.": "neuralops.layers.",
        "neuralops.backward_compat.compression_engine.": "neuralops.compression.",
    }

    def _resolve_canonical(self, canonical_fullname: str) -> Optional[tuple[str, bool, Optional[dict[str, str]]]]:
        parts = canonical_fullname.split(".")
        # Handle neuralops.layers.<leaf>
        if len(parts) == 3 and parts[0] == "neuralops" and parts[1] == "layers":
            leaf = parts[2]
            if leaf == "replace":
                return ("neuralops.core.replace_attention", False, None)
            if leaf in _ATTENTIONS_RENAMED_MAP:
                rel_path, sym_map = _ATTENTIONS_RENAMED_MAP[leaf]
                return (f"neuralops.layers.{rel_path}", False, sym_map)
            if leaf in _ATTENTIONS_LEGACY_MAP:
                rel_path = _ATTENTIONS_LEGACY_MAP[leaf]
                return (f"neuralops.layers.{rel_path}", False, None)

        # Handle neuralops.compression.<leaf>
        if len(parts) == 3 and parts[0] == "neuralops" and parts[1] == "compression":
            leaf = parts[2]
            if leaf in _COMPRESSION_RENAMED_MAP:
                rel_path, sym_map = _COMPRESSION_RENAMED_MAP[leaf]
                return (f"neuralops.compression.{rel_path}", False, sym_map)

        # Direct spec lookup
        try:
            c_spec = importlib.util.find_spec(canonical_fullname)
            if c_spec is not None:
                is_pkg = c_spec.submodule_search_locations is not None
                return (canonical_fullname, is_pkg, None)
        except Exception:
            pass

        # Subdirectory fallback for flat lookups
        if len(parts) == 3 and parts[0] == "neuralops":
            pkg, mod = parts[1], parts[2]
            subpkgs = []
            if pkg == "layers":
                subpkgs = [
                    "linear",
                    "svd",
                    "sparse",
                    "gated",
                    "dense",
                    "ssm",
                    "hybrid",
                    "modern",
                    "adaptive",
                    "fast_kernels",
                    "experimental",
                ]
            elif pkg == "compression":
                subpkgs = ["svd", "quantization", "pruning", "low_rank", "tensor_decomp", "kv_cache", "experimental"]

            for sub in subpkgs:
                candidate = f"neuralops.{pkg}.{sub}.{mod}"
                try:
                    c_spec = importlib.util.find_spec(candidate)
                    if c_spec is not None:
                        is_pkg = c_spec.submodule_search_locations is not None
                        return (candidate, is_pkg, None)
                except Exception:
                    continue
        return None

    def find_spec(
        self,
        fullname: str,
        path: Optional[Sequence[str]],
        target: Optional[ModuleType] = None,
    ) -> Optional[Any]:
        if fullname in self._EXACT_MAP:
            target_name, is_pkg = self._EXACT_MAP[fullname]
            loader = ExactModuleAliasLoader(target_name, is_pkg=is_pkg)
            return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)

        for old_prefix, new_prefix in self._PREFIX_MAP.items():
            if fullname.startswith(old_prefix):
                subpath = fullname[len(old_prefix) :]
                canonical_fullname = f"{new_prefix}{subpath}"
                resolved = self._resolve_canonical(canonical_fullname)
                if resolved is not None:
                    canon, is_pkg, sym_map = resolved
                    loader = ExactModuleAliasLoader(canon, is_pkg=is_pkg, symbol_aliases=sym_map)
                    return importlib.util.spec_from_loader(fullname, loader, is_package=is_pkg)
        return None


def install_compatibility_interceptor() -> None:
    if not any(isinstance(finder, NeuralOpsCompatibilityRedirectFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, NeuralOpsCompatibilityRedirectFinder())
        logger.debug("NeuralOpsCompatibilityRedirectFinder active in sys.meta_path.")
