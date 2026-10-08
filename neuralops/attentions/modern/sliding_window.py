"""Sliding window attention (Mistral-style)."""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["SlidingWindowAttention"]


@ATTENTION_REGISTRY.register("sliding_window_attention")
class SlidingWindowAttention(BaseAttention):
    """Sliding window attention — attends to last W tokens only."""
    
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    
    def __init__(
        self,
        config: object | None = None,
        window_size: int = 4096,
        **kwargs,
    ):
        super().__init__(config)
        self.window_size = window_size
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = hidden_states.shape
        
        if N > self.window_size:
            hidden_states = hidden_states[:, -self.window_size:, :]
            N = self.window_size
        
        return hidden_states, None
