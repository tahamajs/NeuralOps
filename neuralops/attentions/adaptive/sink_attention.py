"""Attention sinks — StreamingLLM style."""
from __future__ import annotations
import torch

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["SinkAttention"]


@ATTENTION_REGISTRY.register("sink_attention")
class SinkAttention(BaseAttention):
    """Attention with dedicated sink tokens that always get attention."""
    
    supports_causal: bool = True
    supports_gqa: bool = True
    supports_rope: bool = True
    
    def __init__(
        self,
        config=None,
        num_sink_tokens: int = 4,
        window_size: int = 4096,
        **kwargs,
    ):
        super().__init__(config)
        self.num_sink_tokens = num_sink_tokens
        self.window_size = window_size
    
    def forward(self, hidden_states: torch.Tensor, attention_mask=None, **kwargs):
        B, N, D = hidden_states.shape
        
        if N <= self.window_size:
            return hidden_states, None
        
        sinks = hidden_states[:, :self.num_sink_tokens, :]
        recent = hidden_states[:, -self.window_size + self.num_sink_tokens:, :]
        combined = torch.cat([sinks, recent], dim=1)
        
        return combined, None
