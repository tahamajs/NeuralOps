"""
CUR-MoE Attention / Compression bridge module.
Re-exports CUR-MoE components from compression_engine and integrates with attention pipeline.
"""

from framework.compression_engine.cur_moe_compressor import (
    CURMoECompressor,
    cur_moe_factory,
    tikhonov_pinv,
)

__all__ = [
    "CURMoECompressor",
    "cur_moe_factory",
    "tikhonov_pinv",
]
