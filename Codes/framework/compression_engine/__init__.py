"""
Compression Engine modules for Efficient Transformer Toolkit.

Provides low-rank factorizations (SVD, AASVD, AdaSVD, Dynamic Entropy SVD, Res-SVD, RPCA, Swift-SVD, Direct SVD, SAES-SVD, MGIE, ERC, Dobi-SVD, Hybrid SVD, SALR, SoKA, LiLLaMA, LSC),
structural pruning and layer shedding (SliceGPT, SparseGPT, ADMM, ARMOR, AST, Bonsai, FLAP, FLAT-LLM, Mamba-Shedder, ProxSparse, SLiM, OATS, LIFTR),
quantization (Astro, FAQ, CoopQ, KMeans, Huffman), and KV cache compression
strategies (DeltaKV, HybridKV, KVTC, CacheNotes, ZeroKV-Neo, MoDeGPT).
"""

import glob
import importlib
import logging
import os

from framework.compression_engine.aa_svd_compressor import AASVDCompressor, SVDLinear
from framework.compression_engine.adasvd_compressor import AdaSVDCompressor
from framework.compression_engine.admm_compressor import ADMMCompressor
from framework.compression_engine.admm_pruner import ADMMPruner
from framework.compression_engine.armor_pruner import ARMORPruner
from framework.compression_engine.ast_pruner import ASTPruner
from framework.compression_engine.astro_quantizer import (
    AstroLinear,
    AstroQuantizer,
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
from framework.compression_engine.bonsai_pruner import BonsaiPruner
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor,
    CacheNotesKVCache,
    CompressionPlanningTokens,
)
from framework.compression_engine.coopq_quantizer import CoopQQuantizer
from framework.compression_engine.delta_kv import DeltaKVCompressor, DeltaKVStore
from framework.compression_engine.direct_svd import DirectSVDCompressor
from framework.compression_engine.dobi_svd import DobiSVDCompressor, DobiSVDLinearLayer
from framework.compression_engine.dynamic_rank import (
    DynamicEntropySVDCompressor,
    DynamicSVDLinear,
    allocate_entropy_ranks,
    compute_singular_entropy,
)
from framework.compression_engine.entropy_budget_allocation import EntropyBudgetCompressor
from framework.compression_engine.erc_svd import ERCCompressor
from framework.compression_engine.faq_quantizer import FAQLinear, FAQQuantizer
from framework.compression_engine.flap import FLAPCompressor
from framework.compression_engine.flat_llm import FLATLLMCompressor
from framework.compression_engine.huffman_coder import (
    HuffmanCoder,
    HuffmanCompressor,
    HuffmanNode,
)
from framework.compression_engine.hybrid_svd_layer import (
    DirectSVDEncoderLayer,
    HybridSVDCompressor,
    SVDSingleHead,
)
from framework.compression_engine.hybridkv_compressor import (
    HybridKVCache,
    HybridKVCompressor,
)
from framework.compression_engine.kmeans_quantize import (
    KMeansCompressor,
    KMeansLinear,
    fast_kmeans_1d,
)
from framework.compression_engine.kvtc_compressor import (
    KVTCCompressor,
    KVTCStore,
    SimpleEntropyCoder,
)
from framework.compression_engine.liftr import LIFTRCompressor
from framework.compression_engine.lillama_compressor import LiLLaMACompressor
from framework.compression_engine.lsc_compressor import LSCCompressor
from framework.compression_engine.mamba_shedder import MambaShedderCompressor
from framework.compression_engine.mgie_svd import MGIECompressor
from framework.compression_engine.mgie_svd_compressor import MGIESVDBudgetCompressor
from framework.compression_engine.modegpt_compressor import MoDeGPTCompressor
from framework.compression_engine.oats_compressor import OATSCompressor
from framework.compression_engine.proxsparse import (
    ProxSparseCompressor,
    ProxSparsePruner,
)
from framework.compression_engine.res_svd import (
    ResSVDCompressor,
    ResSVDLinear,
    SpectralResidualLoRA,
)
from framework.compression_engine.rpca_compressor import RPCACompressor
from framework.compression_engine.saes_svd_v2 import SAESSVDCompressorV2
from framework.compression_engine.salr import SALRCompressor, SALRLayer
from framework.compression_engine.slicegpt import SliceGPTCompressor
from framework.compression_engine.slicegpt_pca import SliceGPTPCACompressor
from framework.compression_engine.slim_compressor import SLiMCompressor
from framework.compression_engine.soka_compressor import (
    LowRankCompressor,
    LowRankLinear,
    SoKACompressor,
)
from framework.compression_engine.sparsegpt import SparseGPTCompressor
from framework.compression_engine.svd_linear import SVDLinearLayer
from framework.compression_engine.swift_svd import SwiftSVDCompressor
from framework.compression_engine.swift_svd_act import SwiftSVDActivationCompressor
from framework.compression_engine.zerokv_neo_compressor import (
    ZeroKVNeoCache,
    ZeroKVNeoCompressor,
)

logger = logging.getLogger(__name__)

# Auto-discover and register all compressor modules in this directory
_pkg_dir = os.path.dirname(__file__)
for _file in glob.glob(os.path.join(_pkg_dir, "*.py")):
    _mod_name = os.path.splitext(os.path.basename(_file))[0]
    if not _mod_name.startswith("_"):
        try:
            importlib.import_module(f"framework.compression_engine.{_mod_name}")
        except Exception as _e:
            logger.debug("Failed to auto-load compressor module %s: %s", _mod_name, _e)

__all__ = [
    # SVD and Spectral Low-Rank Compressors
    "AASVDCompressor",
    "AdaSVDCompressor",
    "DirectSVDCompressor",
    "DobiSVDCompressor",
    "DobiSVDLinearLayer",
    "DynamicEntropySVDCompressor",
    "DynamicSVDLinear",
    "EntropyBudgetCompressor",
    "ERCCompressor",
    "HybridSVDCompressor",
    "DirectSVDEncoderLayer",
    "SVDSingleHead",
    "MGIECompressor",
    "MGIESVDBudgetCompressor",
    "ResSVDCompressor",
    "ResSVDLinear",
    "SpectralResidualLoRA",
    "RPCACompressor",
    "SAESSVDCompressorV2",
    "SAESCompressor",
    "SALRCompressor",
    "SALRLayer",
    "SVDLinear",
    "SVDLinearLayer",
    "SwiftSVDCompressor",
    "SwiftSVDActivationCompressor",
    "allocate_entropy_ranks",
    "compute_singular_entropy",
    "LowRankCompressor",
    "LowRankLinear",
    "SoKACompressor",
    "LiLLaMACompressor",
    "LSCCompressor",
    # Structural Pruning and Layer Shedding
    "SliceGPTCompressor",
    "SliceGPTPCACompressor",
    "SparseGPTCompressor",
    "ADMMCompressor",
    "ADMMPruner",
    "ARMORPruner",
    "ASTPruner",
    "BonsaiPruner",
    "FLAPCompressor",
    "FLATLLMCompressor",
    "MambaShedderCompressor",
    "ProxSparsePruner",
    "ProxSparseCompressor",
    "SLiMCompressor",
    "LIFTRCompressor",
    "OATSCompressor",
    # Quantization
    "AstroQuantizer",
    "AstroLinear",
    "STEFakeQuantize",
    "quantize_dequantize_tensor",
    "FAQQuantizer",
    "FAQLinear",
    "CoopQQuantizer",
    "KMeansCompressor",
    "KMeansLinear",
    "fast_kmeans_1d",
    # KV Cache
    "DeltaKVCompressor",
    "DeltaKVStore",
    "HybridKVCompressor",
    "HybridKVCache",
    "KVTCCompressor",
    "KVTCStore",
    "SimpleEntropyCoder",
    "CacheNotesCompressor",
    "CacheNotesKVCache",
    "CompressionPlanningTokens",
    "ZeroKVNeoCompressor",
    "ZeroKVNeoCache",
    "MoDeGPTCompressor",
    # Huffman
    "HuffmanCompressor",
    "HuffmanCoder",
    "HuffmanNode",
]
