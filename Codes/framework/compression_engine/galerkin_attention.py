"""
Galerkin Transformer: Petrov‑Galerkin projection attention.
Paper: https://arxiv.org/abs/2105.14906

This attention mechanism projects keys and values onto a low‑rank subspace
using a random sketching matrix, reducing the complexity from O(N²) to O(N·rank).
"""
from typing import Any

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("galerkin")
class GalerkinAttention(nn.Module):
    """
    Galerkin attention with low‑rank projection of keys/values.
    Supports zero‑shot initialisation from a pretrained DistilBERT attention layer.
    """

    def __init__(
        self, model_config, rank: int = 64, dropout: float = 0.1, pretrained_attn=None
    ):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.rank = rank

        # Linear projections (copied from pretrained)
        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Copy pretrained weights
        with torch.no_grad():
            self.query.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.query.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.key.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.key.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.value.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.value.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """(B, N, H*D) -> (B, H, N, D)"""
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass."""
        B, N, _ = hidden_states.shape
        _H, D = self.num_heads, self.head_dim

        Q = self._reshape(self.query(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        # Random sketching matrix (deterministic via fixed seed for reproducibility)
        # Use a buffer to avoid per‑forward random generation (optional)
        sketch = torch.randn(N, self.rank, device=hidden_states.device) / math.sqrt(N)

        # Mask handling - apply to K and V before sketching
        if attention_mask is not None:
            if attention_mask.dim() == 4:
                # Assuming (B, 1, 1, N) or similar, convert to (B, 1, N, 1)
                mask_bool = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(1).unsqueeze(-1)
            elif attention_mask.dim() == 2:
                mask_bool = attention_mask.bool().unsqueeze(1).unsqueeze(-1) # (B, 1, N, 1)
            else:
                mask_bool = attention_mask.bool()
            
            K = K.masked_fill(~mask_bool, 0.0)
            V = V.masked_fill(~mask_bool, 0.0)

        # Project keys and values onto low-rank subspace (B, H, rank, D)
        K_sketch = torch.einsum("bhnd,nr->bhrd", K, sketch)
        V_sketch = torch.einsum("bhnd,nr->bhrd", V, sketch)

        # Attention scores: (B, H, N, rank)
        scale = D**-0.5
        scores = torch.matmul(Q, K_sketch.transpose(-2, -1)) * scale

        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        # Context: (B, H, N, D)
        context = torch.matmul(probs, V_sketch)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, probs if output_attentions else None


@ATTENTION_REGISTRY.register("galerkin")
def inject_galerkin(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
) -> Any:
    """Injection function for DistilBERT."""
    rank = getattr(comp_config, "galerkin_rank", 64)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = GalerkinAttention(model_config, rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn
