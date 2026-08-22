"""
SAES-SVD v2: error suppression with fixed rank ratio. Registered as 'saes_svd_v2'.
Alias/re-export module for saes_svd_v2.
"""

from framework.compression_engine.saes_svd_v2 import (
    SAESSVDCompressorV2,
    saes_svd_v2_factory,
)

# Alias for backward compatibility
SAESCompressor = SAESSVDCompressorV2

__all__ = ["SAESSVDCompressorV2", "SAESCompressor", "saes_svd_v2_factory"]
