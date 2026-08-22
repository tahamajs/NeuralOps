"""
Swift-SVD Activation-aware via cross-covariance. Registered as 'swift_svd_act'.
Alias/re-export module for swift_svd_act.
"""

from framework.compression_engine.swift_svd_act import (
    SwiftSVDActivationCompressor,
    swift_svd_act_factory,
)

__all__ = ["SwiftSVDActivationCompressor", "swift_svd_act_factory"]
