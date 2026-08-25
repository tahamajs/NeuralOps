"""
KVTC: KV Cache Transform Coding - PCA decorrelation + adaptive quantization + entropy coding.
Re-exports KVTCCompressor, KVTCStore, and kvtc_factory.
"""
from framework.compression_engine.kvtc_compressor import (
    KVTCCompressor,
    KVTCStore,
    SimpleEntropyCoder,
    kvtc_factory,
)
from framework.core.registry import COMPRESSOR_REGISTRY

COMPRESSOR_REGISTRY.register("kvtc_quantizer")(kvtc_factory)

__all__ = [
    "KVTCCompressor",
    "KVTCStore",
    "SimpleEntropyCoder",
    "kvtc_factory",
]
