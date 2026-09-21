"""
RPCA Compressor alias module.
Re-exports RPCACompressor from rpca_compressor.
"""

from framework.compression_engine.rpca_compressor import (
    RPCACompressor,
    rpca_compressor_factory,
    rpca_factory,
)

__all__ = [
    "RPCACompressor",
    "rpca_compressor_factory",
    "rpca_factory",
]
