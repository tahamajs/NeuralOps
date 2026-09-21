"""
Dobi-SVD alias module.
Re-exports DobiSVDCompressor, DobiSVDLinearLayer, dobi_factory, and injection hooks.
"""

from framework.compression_engine.dobi_svd import (
    DobiSVDLinearLayer,
    DobiSVDCompressor,
    dobi_factory,
    dobi_svd_factory,
    dobi_svd_compressor_factory,
    inject_dobi,
    dobi_inject,
)

__all__ = [
    "DobiSVDLinearLayer",
    "DobiSVDCompressor",
    "dobi_factory",
    "dobi_svd_factory",
    "dobi_svd_compressor_factory",
    "inject_dobi",
    "dobi_inject",
]
