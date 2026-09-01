"""
AASVD (Activation-Aware SVD) Compressor alias module.
Re-exports AASVDCompressor from compression_engine.
"""

from framework.compression_engine.aa_svd_compressor import (
    AASVDCompressor,
    SVDLinear,
    aa_svd_factory,
    aasvd_compressor_factory,
    aasvd_factory,
    DEFAULT_CALIBRATION_CORPUS,
)

__all__ = [
    "AASVDCompressor",
    "SVDLinear",
    "aa_svd_factory",
    "aasvd_compressor_factory",
    "aasvd_factory",
    "DEFAULT_CALIBRATION_CORPUS",
]
