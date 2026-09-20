"""
GreCoDeC Compressor alias module.
Re-exports GreCoDeCCompressor from grecodec_compressor.
"""

from framework.compression_engine.grecodec_compressor import (
    GreCoDeCLinear,
    GreCoDeCCompressor,
    grecodec_compressor_factory,
)

__all__ = [
    "GreCoDeCLinear",
    "GreCoDeCCompressor",
    "grecodec_compressor_factory",
]
