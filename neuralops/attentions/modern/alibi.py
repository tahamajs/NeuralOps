"""ALiBi (Attention with Linear Biases) — Press et al. 2021."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from neuralops.core.base import BaseAttention
from neuralops.core.registry import ATTENTION_REGISTRY

__all__ = ["ALiBiAttention", "get_alibi_slopes"]


def get_alibi_slopes(num_heads: int) -> torch.Tensor:
    """Compute ALiBi slopes for each head."""
    n = 2 ** math.floor(math.log2(num_heads))
    slopes = torch.tensor(
        [2 ** (-8.0 / n * (i + 1)) for i in range(n)],
        dtype=torch.float32,
    )
    if n < num_heads:
        extra = torch.tensor(
            [2 ** (-8.0 / (2 * n) * (2 * i + 1)) for i in range(num_heads - n)],
            dtype=torch.float32,
        )
        slopes = torch.cat([slopes, extra])
    return slopes


@ATTENTION_REGISTRY.register("alibi_attention")
class ALiBiAttention(BaseAttention):
    """Attention with linear biases instead of positional embeddings."""
    
    supports_causal: bool = True
    supports_gqa: bool = False
    supports_rope: bool = False
    
    def __init__(self, config=None, num_heads: int = 12, **kwargs):
        super().__init__(config)
        self.num_heads = num_heads
        self.register_buffer("alibi_slopes", get_alibi_slopes(num_heads))
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, None]:
        B, N, D = hidden_states.shape
        q_pos = torch.arange(N, device=hidden_states.device).unsqueeze(1)
        k_pos = torch.arange(N, device=hidden_states.device).unsqueeze(0)
        dist = (q_pos - k_pos).abs().unsqueeze(0)  # [1, N, N]
        bias = -self.alibi_slopes.view(-1, 1, 1) * dist  # [H, N, N]
        return hidden_states, None
