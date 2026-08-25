"""
AASVD (Activation-Aware SVD) Compressor alias module.
Re-exports AASVDCompressor from aa_svd_compressor.
"""

from framework.compression_engine.aa_svd_compressor import (
    AASVDCompressor,
    SVDLinear,
    aa_svd_factory,
    aasvd_compressor_factory,
)

__all__ = [
    "AASVDCompressor",
    "SVDLinear",
    "aa_svd_factory",
    "aasvd_compressor_factory",
]
