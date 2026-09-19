"""
Naive SVD baseline: plain truncated SVD without calibration, Fisher weighting, or whitening.
Used as an uncalibrated low-rank baseline in comparative evaluations.

Paper comparison reference:
Standard truncated SVD approximation W ~ U[:, :r] @ diag(S[:r]) @ V^T[:r, :].
"""
from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY
from framework.attentions.minimal_svd import MinimalSVDAttention, FactorizedLinear

logger = logging.getLogger(__name__)


@ATTENTION_REGISTRY.register("naive_svd", supports_causal=True, supports_gqa=True)
class NaiveSVDAttention(MinimalSVDAttention):
    r"""
    Naive truncated SVD baseline attention module.
    Inherits production-hardened multi-architecture routing from MinimalSVDAttention
    with uncalibrated plain truncated SVD initialization and no recovery adapters.
    """
    is_custom_attention: bool = True
    supports_causal: bool = True
    supports_gqa: bool = True

    def __init__(
        self,
        config: Optional[Any] = None,
        model_config: Optional[Any] = None,
        energy: float = 1.0,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        lora_rank: int = 0,  # Zero recovery parameters for naive baseline
        lora_alpha: float = 1.0,
        is_decoder: Optional[bool] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            config=config,
            model_config=model_config,
            energy=energy,
            dropout=dropout,
            pretrained_attn=pretrained_attn,
            rank=rank,
            lora_rank=0,
            lora_alpha=1.0,
            is_decoder=is_decoder,
            **kwargs,
        )


@ATTENTION_REGISTRY.register_inject("naive_svd")
def inject_naive_svd(
    layer: nn.Module,
    comp_config: Any = None,
    old_attn: Optional[nn.Module] = None,
    model_config: Optional[Any] = None,
) -> Optional[nn.Module]:
    """Factory hook to instantiate and inject NaiveSVDAttention into transformer layers."""
    rank = getattr(comp_config, "rank", 64) if comp_config else 64
    new_attn = NaiveSVDAttention(
        model_config=model_config,
        rank=rank,
        energy=1.0,
        dropout=getattr(comp_config, "dropout", 0.1) if comp_config else 0.1,
    )
    src_attn = old_attn if old_attn is not None else getattr(layer, "attention", getattr(layer, "self", None))
    if src_attn is not None:
        try:
            new_attn.copy_weights(src_attn)
        except Exception as e:
            logger.error("copy_weights failed for naive_svd: %s", e)
            return None
        p = next(src_attn.parameters(), None)
        if p is not None:
            new_attn.to(device=p.device, dtype=p.dtype)

    if hasattr(layer, "attention"):
        if hasattr(layer.attention, "self"):
            layer.attention.self = new_attn
            new_attn.is_nested_encoder = True
        else:
            layer.attention = new_attn
            new_attn.is_nested_encoder = False
    elif hasattr(layer, "self"):
        layer.self = new_attn
        new_attn.is_nested_encoder = True
    else:
        new_attn.is_nested_encoder = False
    return new_attn
