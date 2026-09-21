"""
CUR-MoE Compressor alias module.
Re-exports CURMoECompressor from cur_moe_compressor.
"""

from framework.compression_engine.cur_moe_compressor import (
    CURMoECompressor,
    cur_moe_factory,
    inject_cur_moe,
    cur_moe_inject,
    tikhonov_pinv,
)

__all__ = [
    "CURMoECompressor",
    "cur_moe_factory",
    "inject_cur_moe",
    "cur_moe_inject",
    "tikhonov_pinv",
]
