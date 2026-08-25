"""
FRanDI Compressor alias module.
Re-exports FRanDICompressor from frandi_compressor.
"""

from framework.compression_engine.frandi_compressor import (
    FRanDILinear,
    FRanDICompressor,
    frandi_factory,
)

__all__ = [
    "FRanDILinear",
    "FRanDICompressor",
    "frandi_factory",
]
