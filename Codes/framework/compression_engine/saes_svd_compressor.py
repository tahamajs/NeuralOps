"""
SAES-SVD Compressor: Joint intra- and inter-layer error compensation. Registered as 'saes'.
Alias/re-export module for saes_svd_v2.
"""

from framework.compression_engine.saes_svd_v2 import (
    SAESSVDCompressorV2,
    saes_svd_v2_factory,
)

SAESCompressor = SAESSVDCompressorV2

__all__ = ["SAESCompressor", "SAESSVDCompressorV2", "saes_svd_v2_factory"]
