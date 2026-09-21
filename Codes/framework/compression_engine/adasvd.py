"""
AdaSVD alias shim: maps framework.compression_engine.adasvd to adasvd_compressor.
"""

from framework.compression_engine.adasvd_compressor import (
    AdaSVDCompressor,
    adasvd_factory,
    adasvd_compressor_factory,
)

__all__ = ["AdaSVDCompressor", "adasvd_factory", "adasvd_compressor_factory"]
