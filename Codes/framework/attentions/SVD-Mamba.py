"""Compatibility re-export for SVD-Mamba from svd_mamba."""
from framework.attentions.svd_mamba import (
    SVDMamba,
    SVDLinear,
    inject_svd_mamba,
    inject_svd_mamba_attention,
    inject_SVD_Mamba,
)

__all__ = [
    "SVDMamba",
    "SVDLinear",
    "inject_svd_mamba",
    "inject_svd_mamba_attention",
    "inject_SVD_Mamba",
]

