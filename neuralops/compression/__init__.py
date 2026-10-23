#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Unified Compression Engines (Tier 2).
"""

from __future__ import annotations

import importlib
import sys
import types
from importlib.abc import Loader, MetaPathFinder
from importlib.util import spec_from_loader
from typing import Any

from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
from neuralops.core.registry import COMPRESSOR_REGISTRY, get_compressor, list_compressors

COMPRESSOR_REGISTRY = COMPRESSOR_REGISTRY
get_compressor = get_compressor
list_compressors = list_compressors
BaseCompressor = BaseCompressor
compress_model = compress_model

_SUBPACKAGES = [
    'experimental', 'kv_cache', 'low_rank', 'pruning', 'quantization', 'svd', 'tensor_decomp'
]

_LEGACY_MAP = {
    'hadamard_incoherent_svd': 'svd.hadamard_incoherent_svd', 'hadamard_svd': 'svd.hadamard_incoherent_svd',
    'liftr': 'experimental.liftr', 'lillama_compressor': 'experimental.lillama_compressor',
    'flat_llm': 'experimental.flat_llm', 'hashed_net_compressor': 'experimental.hashed_net_compressor',
    'roast_hashed_linear': 'experimental.roast_hashed_linear', 'aatc_xkv_compressor': 'kv_cache.aatc_xkv_compressor',
    'delta_kv': 'kv_cache.delta_kv', 'kvtc_compressor': 'kv_cache.kvtc_compressor',
    'hybridkv_compressor': 'kv_cache.hybridkv_compressor', 'cachenotes_compressor': 'kv_cache.cachenotes_compressor',
    'zerokv_neo_compressor': 'kv_cache.zerokv_neo_compressor', 'rpca_compressor': 'low_rank.rpca_compressor',
    'frandi_compressor': 'low_rank.frandi_compressor', 'sdrm_compressor': 'low_rank.sdrm_compressor',
    'mgie_svd_compressor': 'low_rank.mgie_svd_compressor', 'cr_compressor': 'low_rank.cr_compressor',
    'cur_compressor': 'low_rank.cur_compressor', 'grecodec_compressor': 'low_rank.grecodec_compressor',
    'lsc_compressor': 'low_rank.lsc_compressor', 'drone_compressor': 'low_rank.drone_compressor',
    'oats_compressor': 'low_rank.oats_compressor', 'salr': 'low_rank.salr',
    'uniql_compressor': 'low_rank.uniql_compressor', 'cur_moe_compressor': 'low_rank.cur_moe_compressor',
    'soka_compressor': 'low_rank.soka_compressor', 'tomoe_compressor': 'low_rank.tomoe_compressor',
    'compot_compressor': 'low_rank.compot_compressor', 'come_compressor': 'low_rank.come_compressor',
    'modegpt_compressor': 'low_rank.modegpt_compressor', 'enlighten_compressor': 'low_rank.enlighten_compressor',
    'a3_compressor': 'low_rank.a3_compressor', 'slim_compressor': 'low_rank.slim_compressor',
    'nmf_compressor': 'low_rank.nmf_compressor', 'mlorq_compressor': 'low_rank.mlorq_compressor',
    'svd_llm_compressor': 'low_rank.svd_llm_compressor', 'admm_compressor': 'pruning.admm_compressor',
    'mamba_shedder': 'pruning.mamba_shedder', 'ast_pruner': 'pruning.ast_pruner',
    'flap': 'pruning.flap', 'proxsparse': 'pruning.proxsparse',
    'bonsai_pruner': 'pruning.bonsai_pruner', 'sparsegpt': 'pruning.sparsegpt',
    'armor_pruner': 'pruning.armor_pruner', 'admm_pruner': 'pruning.admm_pruner',
    'coopq_quantizer': 'quantization.coopq_quantizer', 'slicegpt': 'quantization.slicegpt',
    'faq_quantizer': 'quantization.faq_quantizer', 'rotational_slicegpt_int4': 'quantization.rotational_slicegpt_int4',
    'kmeans_quantize': 'quantization.kmeans_quantize', 'slicegpt_pca': 'quantization.slicegpt_pca',
    'huffman_coder': 'quantization.huffman_coder', 'astro_quantizer': 'quantization.astro_quantizer',
    'dynamic_rank': 'svd.dynamic_rank', 'saes_svd_v2': 'svd.saes_svd_v2',
    'optimal_subspace_projection': 'svd.optimal_subspace_projection', 'gfws_d_compressor': 'svd.gfws_d_compressor',
    'fwsvd_compressor': 'svd.fwsvd_compressor', 'swift_svd': 'svd.swift_svd',
    'erc_svd': 'svd.erc_svd', 'aa_svd_compressor': 'svd.aa_svd_compressor',
    'subspace_projection': 'svd.subspace_projection', 'entropy_budget_allocation': 'svd.entropy_budget_allocation',
    'svd_linear': 'svd.svd_linear', 'swift_svd_act': 'svd.swift_svd_act',
    'hybrid_svd_layer': 'svd.hybrid_svd_layer', 'direct_svd': 'svd.direct_svd',
    'adaptive_ranks': 'svd.adaptive_ranks', 'residual_svd_compressor': 'svd.residual_svd_compressor',
    'dobi_svd': 'svd.dobi_svd', 'adasvd_compressor': 'svd.adasvd_compressor',
    'kronecker_compressor': 'tensor_decomp.kronecker_compressor', 'tt_compressor': 'tensor_decomp.tt_compressor',
    'kronecker_tt_lora': 'tensor_decomp.kronecker_tt_lora'
}

_RENAMED_MODULES = {
    'res_svd': ('svd.residual_svd_compressor', {'ResSVDCompressor': 'ResidualSVDCompressor', 'inject_res_svd': 'inject_residual_svd_compressor'}),
}


def _get_target_module(leaf: str):
    if leaf in _RENAMED_MODULES:
        rel_path, class_map = _RENAMED_MODULES[leaf]
        target_mod = importlib.import_module(f".{rel_path}", __package__)
        alias_mod = types.ModuleType(f"{__name__}.{leaf}")
        alias_mod.__dict__.update(target_mod.__dict__)
        for old_k, target_k in class_map.items():
            if hasattr(target_mod, target_k):
                setattr(alias_mod, old_k, getattr(target_mod, target_k))
        return alias_mod
    elif leaf in _LEGACY_MAP:
        rel_path = _LEGACY_MAP[leaf]
        return importlib.import_module(f".{rel_path}", __package__)
    return None


class _LegacyCompLoader(Loader):
    def __init__(self, leaf: str, fullname: str):
        self.leaf = leaf
        self.fullname = fullname

    def create_module(self, _spec):
        target_mod = _get_target_module(self.leaf)
        if target_mod is not None:
            sys.modules[self.fullname] = target_mod
            return target_mod
        mod = types.ModuleType(self.fullname)
        sys.modules[self.fullname] = mod
        return mod

    def exec_module(self, module):
        pass


class _LegacyCompFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(__name__ + "."):
            leaf = fullname[len(__name__) + 1:]
            if leaf in _LEGACY_MAP or leaf in _RENAMED_MODULES:
                return spec_from_loader(fullname, _LegacyCompLoader(leaf, fullname))
        return None


sys.meta_path.insert(0, _LegacyCompFinder())


def __getattr__(name: str) -> Any:
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute {name!r}")
    if name in _SUBPACKAGES:
        module = importlib.import_module(f".{name}", __package__)
        globals()[name] = module
        return module
    if name in _LEGACY_MAP or name in _RENAMED_MODULES:
        mod = _get_target_module(name)
        if mod is not None:
            globals()[name] = mod
            return mod
    for subpackage in _SUBPACKAGES:
        try:
            module = importlib.import_module(f".{subpackage}", __package__)
            if hasattr(module, name):
                val = getattr(module, name)
                globals()[name] = val
                return val
        except (ImportError, AttributeError):
            continue
    raise AttributeError(f"module '{__name__}' has no attribute {name!r}")


def __dir__() -> list[str]:
    names = (
        set(globals())
        | set(_SUBPACKAGES)
        | set(_LEGACY_MAP)
        | set(_RENAMED_MODULES)
        | {"COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model", "get_compressor", "list_compressors"}
    )
    for subpackage in _SUBPACKAGES:
        try:
            mod = importlib.import_module(f".{subpackage}", __package__)
            names.update(getattr(mod, "__all__", []))
        except Exception:
            pass
    return sorted(names)


__all__ = sorted(list(set(__dir__())))
