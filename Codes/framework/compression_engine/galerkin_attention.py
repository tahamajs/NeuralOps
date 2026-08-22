"""
Galerkin Transformer: Petrov‑Galerkin projection attention.
Fixes:
- Use a persistent sketch buffer (deterministic per sequence length).
- Unified mask handling (bool/0-1/additive).
- Separate injection registration to avoid duplicate name.
"""
from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("galerkin")
class GalerkinAttention(nn.Module):
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

        self.query = nn.Linear(self.hidden_size, self.all_head_size)
        self.key = nn.Linear(self.hidden_size, self.all_head_size)
        self.value = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.query.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.query.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.key.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.key.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.value.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.value.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # Persistent sketch buffer; will be generated on first forward for current N
        self.register_buffer("sketch", None)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _get_sketch(self, N: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.sketch is None or self.sketch.shape[0] != N:
            # Generate deterministic sketch for this sequence length
            sketch = torch.randn(N, self.rank, device=device, dtype=dtype) / math.sqrt(N)
            self.sketch = sketch
        return self.sketch

    def _convert_mask_to_bool(self, attention_mask, B, N, device):
        """Convert various mask formats to a boolean valid mask of shape [B, N]."""
        if attention_mask is None:
            return torch.ones(B, N, device=device, dtype=torch.bool)
        if attention_mask.dim() == 2:
            # Could be 0/1 float or bool
            if attention_mask.dtype == torch.bool:
                return attention_mask
            else:
                return attention_mask > 0
        elif attention_mask.dim() == 4:
            # Assume additive or 0/1; we'll detect
            if attention_mask.dtype == torch.bool:
                return attention_mask[:, 0, 0, :]
            else:
                # If values are 0 and -inf/-10000, then valid where > -inf
                if attention_mask.min() < -1000:
                    return attention_mask[:, 0, 0, :] > -10000.0
                else:
                    # Assume 0/1
                    return attention_mask[:, 0, 0, :] > 0
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        output_attentions=False,
        **kwargs
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self._reshape(self.query(hidden_states), B)   # (B,H,N,D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        # Convert mask to boolean [B,N]
        valid_mask = self._convert_mask_to_bool(attention_mask, B, N, hidden_states.device)
        # Expand for head and feature dims: [B,1,N,1]
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(hidden_states.dtype)
        K = K * mask_expanded
        V = V * mask_expanded

        sketch = self._get_sketch(N, hidden_states.device, hidden_states.dtype)

        # Project keys and values: (B,H,N,rank)
        K_sketch = torch.einsum("bhnd,nr->bhrd", K, sketch)
        V_sketch = torch.einsum("bhnd,nr->bhrd", V, sketch)

        scale = D ** -0.5
        scores = torch.matmul(Q, K_sketch.transpose(-2, -1)) * scale   # (B,H,N,rank)
        probs = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context = torch.matmul(probs, V_sketch)   # (B,H,N,D)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, probs if output_attentions else None


@ATTENTION_REGISTRY.register("galerkin_inject")
def inject_galerkin(layer, model_config, comp_config, layer_idx, total_layers, old_attn):
    rank = getattr(comp_config, "galerkin_rank", 64)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = GalerkinAttention(model_config, rank, dropout, old_attn)
    layer.attention = new_attn
    return new_attn