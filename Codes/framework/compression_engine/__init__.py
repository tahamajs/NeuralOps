"""
Compression Engine modules for Efficient Transformer Toolkit.

Provides low-rank factorizations (SVD, AASVD, Dynamic Entropy SVD, Res-SVD),
quantization (Astro, FAQ, CoopQ, KMeans, Huffman), and KV cache compression
strategies (DeltaKV, HybridKV, KVTC, CacheNotes, ZeroKV-Neo, MoDeGPT).
"""

import glob
import importlib
import logging
import os

from framework.compression_engine.astro_quantizer import (
    AstroLinear,
    AstroQuantizer,
    STEFakeQuantize,
    quantize_dequantize_tensor,
)
from framework.compression_engine.cachenotes_compressor import (
    CacheNotesCompressor,
    CacheNotesKVCache,
    CompressionPlanningTokens,
)
from framework.compression_engine.coopq_quantizer import CoopQQuantizer
from framework.compression_engine.delta_kv import DeltaKVCompressor, DeltaKVStore
from framework.compression_engine.dynamic_rank import (
    DynamicEntropySVDCompressor,
    DynamicSVDLinear,
    allocate_entropy_ranks,
    compute_singular_entropy,
)
from framework.compression_engine.faq_quantizer import FAQLinear, FAQQuantizer
from framework.compression_engine.huffman_coder import (
    HuffmanCoder,
    HuffmanCompressor,
    HuffmanNode,
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
from framework.compression_engine.modegpt_compressor import MoDeGPTCompressor
from framework.compression_engine.res_svd import (
    ResSVDCompressor,
    ResSVDLinear,
    SpectralResidualLoRA,
)
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
    "DynamicEntropySVDCompressor",
    "DynamicSVDLinear",
    "compute_singular_entropy",
    "allocate_entropy_ranks",
    "ResSVDCompressor",
    "ResSVDLinear",
    "SpectralResidualLoRA",
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
    "HuffmanCompressor",
    "HuffmanCoder",
    "HuffmanNode",
]

