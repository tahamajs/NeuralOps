"""
CURE: CUR-MoE Compressor alias and factory.
Compresses Mixture of Experts (MoE) layers and dense FFN layers using
proper factorized CURLinear (C @ U @ R parameter reduction).
Registered in COMPRESSOR_REGISTRY as 'cure' and 'cure_compressor'.
"""

from __future__ import annotations

from typing import Any, Optional

import torch.nn as nn
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.compression_engine.cur_moe_compressor import CURMoECompressor, cur_moe_factory


@COMPRESSOR_REGISTRY.register("cure")
@COMPRESSOR_REGISTRY.register("cure_compressor")
def cure_factory(model: nn.Module, comp_config: Optional[Any] = None, **kwargs: Any) -> Any:
    """
    CURE compressor factory that delegates to the fully factorized CURMoECompressor,
    ensuring real parameter reduction via factorized CURLinear modules.
    """
    if "rank" not in kwargs and comp_config is not None:
        rank = getattr(comp_config, "cure_rank", getattr(comp_config, "rank", 64))
        kwargs["rank"] = rank
    return cur_moe_factory(model, comp_config=comp_config, **kwargs)
