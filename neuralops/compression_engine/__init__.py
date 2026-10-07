"""
Auto-import all compression_engine modules across subpackages and export their registered classes.
"""

import importlib
import logging
import os
import sys

from neuralops.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

__all__ = ["COMPRESSOR_REGISTRY"]

_SUBPACKAGES = ['experimental', 'kv_cache', 'low_rank', 'pruning', 'quantization', 'svd', 'tensor_decomp']
_LEGACY_MAP = {'liftr': 'experimental.liftr', 'lillama_compressor': 'experimental.lillama_compressor', 'flat_llm': 'experimental.flat_llm', 'hashed_net_compressor': 'experimental.hashed_net_compressor', 'roast_hashed_linear': 'experimental.roast_hashed_linear', 'aatc_xkv_compressor': 'kv_cache.aatc_xkv_compressor', 'delta_kv': 'kv_cache.delta_kv', 'kvtc_compressor': 'kv_cache.kvtc_compressor', 'hybridkv_compressor': 'kv_cache.hybridkv_compressor', 'cachenotes_compressor': 'kv_cache.cachenotes_compressor', 'zerokv_neo_compressor': 'kv_cache.zerokv_neo_compressor', 'rpca_compressor': 'low_rank.rpca_compressor', 'frandi_compressor': 'low_rank.frandi_compressor', 'sdrm_compressor': 'low_rank.sdrm_compressor', 'mgie_svd_compressor': 'low_rank.mgie_svd_compressor', 'cr_compressor': 'low_rank.cr_compressor', 'cur_compressor': 'low_rank.cur_compressor', 'grecodec_compressor': 'low_rank.grecodec_compressor', 'lsc_compressor': 'low_rank.lsc_compressor', 'drone_compressor': 'low_rank.drone_compressor', 'oats_compressor': 'low_rank.oats_compressor', 'salr': 'low_rank.salr', 'uniql_compressor': 'low_rank.uniql_compressor', 'cur_moe_compressor': 'low_rank.cur_moe_compressor', 'soka_compressor': 'low_rank.soka_compressor', 'tomoe_compressor': 'low_rank.tomoe_compressor', 'compot_compressor': 'low_rank.compot_compressor', 'come_compressor': 'low_rank.come_compressor', 'modegpt_compressor': 'low_rank.modegpt_compressor', 'enlighten_compressor': 'low_rank.enlighten_compressor', 'a3_compressor': 'low_rank.a3_compressor', 'slim_compressor': 'low_rank.slim_compressor', 'nmf_compressor': 'low_rank.nmf_compressor', 'mlorq_compressor': 'low_rank.mlorq_compressor', 'svd_llm_compressor': 'low_rank.svd_llm_compressor', 'admm_compressor': 'pruning.admm_compressor', 'mamba_shedder': 'pruning.mamba_shedder', 'ast_pruner': 'pruning.ast_pruner', 'flap': 'pruning.flap', 'proxsparse': 'pruning.proxsparse', 'bonsai_pruner': 'pruning.bonsai_pruner', 'sparsegpt': 'pruning.sparsegpt', 'armor_pruner': 'pruning.armor_pruner', 'admm_pruner': 'pruning.admm_pruner', 'coopq_quantizer': 'quantization.coopq_quantizer', 'slicegpt': 'quantization.slicegpt', 'faq_quantizer': 'quantization.faq_quantizer', 'rotational_slicegpt_int4': 'quantization.rotational_slicegpt_int4', 'kmeans_quantize': 'quantization.kmeans_quantize', 'slicegpt_pca': 'quantization.slicegpt_pca', 'huffman_coder': 'quantization.huffman_coder', 'astro_quantizer': 'quantization.astro_quantizer', 'dynamic_rank': 'svd.dynamic_rank', 'saes_svd_v2': 'svd.saes_svd_v2', 'optimal_subspace_projection': 'svd.optimal_subspace_projection', 'gfws_d_compressor': 'svd.gfws_d_compressor', 'fwsvd_compressor': 'svd.fwsvd_compressor', 'swift_svd': 'svd.swift_svd', 'erc_svd': 'svd.erc_svd', 'aa_svd_compressor': 'svd.aa_svd_compressor', 'subspace_projection': 'svd.subspace_projection', 'entropy_budget_allocation': 'svd.entropy_budget_allocation', 'svd_linear': 'svd.svd_linear', 'swift_svd_act': 'svd.swift_svd_act', 'hybrid_svd_layer': 'svd.hybrid_svd_layer', 'direct_svd': 'svd.direct_svd', 'adaptive_ranks': 'svd.adaptive_ranks', 'residual_svd_compressor': 'svd.residual_svd_compressor', 'dobi_svd': 'svd.dobi_svd', 'adasvd_compressor': 'svd.adasvd_compressor', 'kronecker_compressor': 'tensor_decomp.kronecker_compressor', 'tt_compressor': 'tensor_decomp.tt_compressor', 'kronecker_tt_lora': 'tensor_decomp.kronecker_tt_lora'}

# 1. Backward compatibility shim for sys.modules and import hooks (old flat import paths -> new subpackages)
from importlib.abc import MetaPathFinder, Loader
from importlib.util import spec_from_loader

class _LegacyCompLoader(Loader):
    def __init__(self, target_name):
        self.target_name = target_name
    def create_module(self, spec):
        return importlib.import_module(self.target_name)
    def exec_module(self, module):
        pass

class _LegacyCompFinder(MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname.startswith(__name__ + "."):
            leaf = fullname[len(__name__) + 1:]
            if leaf in _LEGACY_MAP:
                target_name = f"{__name__}.{_LEGACY_MAP[leaf]}"
                return spec_from_loader(fullname, _LegacyCompLoader(target_name))
        return None

sys.meta_path.insert(0, _LegacyCompFinder())

# Pre-populate sys.modules aliases
for _old, _new in _LEGACY_MAP.items():
    try:
        _target_mod = importlib.import_module(f".{_new}", __package__)
        sys.modules[f"{__name__}.{_old}"] = _target_mod
    except Exception as _e:
        logger.debug(f"Initial legacy mapping deferred for {_old}: {_e}")

# 2. Import all subpackages and re-export classes at top level
for _sub in _SUBPACKAGES:
    try:
        _submod = importlib.import_module(f".{_sub}", __package__)
        for _attr in dir(_submod):
            if not _attr.startswith("_"):
                _val = getattr(_submod, _attr)
                if isinstance(_val, type):
                    globals()[_attr] = _val
                    if _attr not in __all__:
                        __all__.append(_attr)
    except Exception as _e:
        logger.debug(f"Failed importing subpackage {_sub}: {_e}")
