"""
Compression Engine modules for Efficient Transformer Toolkit.

Provides low-rank factorizations (SVD, AASVD, AdaSVD, Dynamic Entropy SVD, Res-SVD, RPCA, Swift-SVD, Direct SVD, SAES-SVD, MGIE, ERC, Dobi-SVD, Hybrid SVD, SALR, SoKA, LiLLaMA, LSC),
structural pruning and layer shedding (SliceGPT, SparseGPT, ADMM, ARMOR, AST, Bonsai, FLAP, FLAT-LLM, Mamba-Shedder, ProxSparse, SLiM, OATS, LIFTR),
quantization (Astro, FAQ, CoopQ, KMeans, Huffman), and KV cache compression
strategies (DeltaKV, HybridKV, KVTC, CacheNotes, ZeroKV-Neo, MoDeGPT).
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)

# Map symbol names to their module filename within framework.compression_engine
_MODULE_MAP: Dict[str, str] = {
    # SVD and Spectral Low-Rank Compressors
    "AASVDCompressor": "aa_svd_compressor",
    "SVDLinear": "aa_svd_compressor",
    "AdaptiveTransformerCompressor": "adaptive_compressor",
    "AdaptiveRanksCompressor": "adaptive_ranks",
    "AdaSVDCompressor": "adasvd_compressor",
    "DirectSVDCompressor": "direct_svd",
    "DobiSVDCompressor": "dobi_svd",
    "DobiSVDLinearLayer": "dobi_svd",
    "DynamicEntropySVDCompressor": "dynamic_rank",
    "DynamicSVDLinear": "dynamic_rank",
    "allocate_entropy_ranks": "dynamic_rank",
    "compute_singular_entropy": "dynamic_rank",
    "EntropyBudgetCompressor": "entropy_budget_allocation",
    "ERCCompressor": "erc_svd",
    "HybridSVDCompressor": "hybrid_svd_layer",
    "DirectSVDEncoderLayer": "hybrid_svd_layer",
    "SVDSingleHead": "hybrid_svd_layer",
    "MGIECompressor": "mgie_svd",
    "MGIESVDBudgetCompressor": "mgie_svd_compressor",
    "ResSVDCompressor": "res_svd",
    "ResSVDLinear": "res_svd",
    "SpectralResidualLoRA": "res_svd",
    "RPCACompressor": "rpca_compressor",
    "SAESSVDCompressorV2": "saes_svd_v2",
    "SAESCompressor": "saes_svd",
    "SALRCompressor": "salr",
    "SALRLayer": "salr",
    "SDRMCompressor": "sdrm_compressor",
    "SVDLinearLayer": "svd_linear",
    "SVDLinearCompressor": "svd_linear",
    "SwiftSVDCompressor": "swift_svd",
    "SwiftSVDActivationCompressor": "swift_svd_act",
    "UniQLCompressor": "uniql_compressor",
    "LowRankCompressor": "soka_compressor",
    "LowRankLinear": "soka_compressor",
    "SoKACompressor": "soka_compressor",
    "soka_factory": "soka_compressor",
    "LiLLaMACompressor": "lillama_compressor",
    "lillama_factory": "lillama_compressor",
    "LSCCompressor": "lsc_compressor",
    "lsc_factory": "lsc_compressor",
    # Structural Pruning and Layer Shedding
    "SliceGPTCompressor": "slicegpt",
    "slicegpt_factory": "slicegpt",
    "SliceGPTPCACompressor": "slicegpt_pca",
    "slicegpt_pca_factory": "slicegpt_pca",
    "RotationalSliceGPTINT4Compressor": "rotational_slicegpt_int4",
    "RotationalSliceGPTCompressor": "rotational_slicegpt_int4",
    "RotationalSliceGPTINT4Linear": "rotational_slicegpt_int4",
    "rotational_slicegpt_int4_factory": "rotational_slicegpt_int4",
    "OptimalSubspaceProjectionCompressor": "optimal_subspace_projection",
    "SubspaceProjectionCompressor": "optimal_subspace_projection",
    "optimal_subspace_projection_factory": "optimal_subspace_projection",
    "SubspaceLinear": "subspace_projection",
    "SparseGPTCompressor": "sparsegpt",
    "sparsegpt_factory": "sparsegpt",
    "ADMMCompressor": "admm_compressor",
    "admm_factory": "admm_compressor",
    "ADMMPruner": "admm_pruner",
    "admm_pruner_factory": "admm_pruner",
    "ARMORPruner": "armor_pruner",
    "armor_pruner_factory": "armor_pruner",
    "ASTPruner": "ast_pruner",
    "ast_pruner_factory": "ast_pruner",
    "BonsaiPruner": "bonsai_pruner",
    "bonsai_factory": "bonsai_pruner",
    "ENLightenCompressor": "enlighten_compressor",
    "FLAPCompressor": "flap",
    "flap_compressor_factory": "flap",
    "FLATLLMCompressor": "flat_llm",
    "flat_llm_factory": "flat_llm",
    "MambaShedderCompressor": "mamba_shedder",
    "mamba_shedder_factory": "mamba_shedder",
    "ProxSparsePruner": "proxsparse",
    "ProxSparseCompressor": "proxsparse",
    "proxsparse_factory": "proxsparse",
    "SLiMCompressor": "slim_compressor",
    "slim_factory": "slim_compressor",
    "LIFTRCompressor": "liftr",
    "liftr_compressor_entry": "liftr",
    "liftr_factory": "liftr",
    "OATSCompressor": "oats_compressor",
    "oats_compressor_factory": "oats_compressor",
    "CoMeCompressor": "come",
    "CoMeBridge": "come",
    "CoMeBridgedLayer": "come",
    "come_factory": "come",
    # Quantization
    "AstroQuantizer": "astro_quantizer",
    "AstroLinear": "astro_quantizer",
    "STEFakeQuantize": "astro_quantizer",
    "quantize_dequantize_tensor": "astro_quantizer",
    "FAQQuantizer": "faq_quantizer",
    "FAQLinear": "faq_quantizer",
    "CoopQQuantizer": "coopq_quantizer",
    "KMeansCompressor": "kmeans_quantize",
    "KMeansLinear": "kmeans_quantize",
    "fast_kmeans_1d": "kmeans_quantize",
    # KV Cache
    "DeltaKVCompressor": "delta_kv",
    "DeltaKVStore": "delta_kv",
    "HybridKVCompressor": "hybridkv_compressor",
    "HybridKVCache": "hybridkv_compressor",
    "KVTCCompressor": "kvtc_compressor",
    "KVTCStore": "kvtc_compressor",
    "SimpleEntropyCoder": "kvtc_compressor",
    "CacheNotesCompressor": "cachenotes_compressor",
    "CacheNotesKVCache": "cachenotes_compressor",
    "CompressionPlanningTokens": "cachenotes_compressor",
    "ZeroKVNeoCompressor": "zerokv_neo_compressor",
    "ZeroKVNeoCache": "zerokv_neo_compressor",
    "MoDeGPTCompressor": "modegpt_compressor",
    # Huffman
    "HuffmanCompressor": "huffman_coder",
    "HuffmanCoder": "huffman_coder",
    "HuffmanNode": "huffman_coder",
    # NLI Softmax
    "NLISoftmaxFunction": "nli_softmax",
    "nli_softmax": "nli_softmax",
    "NLISoftmax": "nli_softmax",
    "NLISoftmaxCompressor": "nli_softmax",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for submodules and compressor classes."""
    if name in _MODULE_MAP:
        mod_name = _MODULE_MAP[name]
        try:
            module = importlib.import_module(f"framework.compression_engine.{mod_name}")
            val = getattr(module, name)
            globals()[name] = val
            return val
        except Exception as e:
            logger.error("Failed to lazily load %s from %s: %s", name, mod_name, e)
            raise
    # Submodule lookup fallback
    try:
        module = importlib.import_module(f"framework.compression_engine.{name}")
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(_MODULE_MAP.keys())

# Auto-discover and register all compressor modules in this package
import glob
import os

_pkg_dir = os.path.dirname(__file__)
for _file in glob.glob(os.path.join(_pkg_dir, "*.py")):
    _mod_name = os.path.splitext(os.path.basename(_file))[0]
    if not _mod_name.startswith("_"):
        try:
            importlib.import_module(f"framework.compression_engine.{_mod_name}")
        except Exception as _e:
            logger.debug("Failed to auto-load compressor module %s: %s", _mod_name, _e)
