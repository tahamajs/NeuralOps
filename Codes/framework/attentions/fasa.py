"""
Frequency/Variance-Aware Sparse Attention (FASA) / MixAttention-MoSKA Module.
"""

from typing import Any, Optional
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.mixattention_moska import MixAttentionMoSKA


@ATTENTION_REGISTRY.register("fasa")
class FASAAttention(MixAttentionMoSKA):
    r"""
    FASA: Frequency/Variance-Aware Sparse Attention.
    Inherits directly from MixAttentionMoSKA with full [CLS] token protection and variance-based pruning.
    """
    pass
