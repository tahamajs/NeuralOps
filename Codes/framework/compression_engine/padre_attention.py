"""
PADRe – Polynomial Attention with Dropout Regularization.
Replaces softmax with a stable polynomial approximation, enabling faster
inference and potential for linear‑time attention via kernelisation.
"""
from typing import Any

import torch
import torch.nn as nn
from framework.core.base import BaseAttention


class PADReAttention(BaseAttention):
    """
    Polynomial Attention with Dropout Regularization (PADRe).

    Args:
        config: model configuration (HuggingFace style)
        order: degree of the polynomial kernel (higher = closer to softmax)
        temperature: scaling factor applied before polynomial
    """

    def __init__(self, config, order: int = 2, temperature: float = 1.0, **kwargs):
        super().__init__(config, rank=order)  # rank used for order tracking
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.order = order
        self.temperature = temperature

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Reshape to (batch, heads, seq_len, head_dim)."""
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor = None,
        head_mask: torch.Tensor = None,
        output_attentions: bool = False,
        **kwargs,
    ) -> Any:
        """Forward pass."""
        B, N, _ = hidden_states.shape
        Q = self._reshape(self.query(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.key(hidden_states), B)
        V = self._reshape(self.value(hidden_states), B)

        scale = 1.0 / (self.head_dim**0.5)
        # raw scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Apply temperature and re‑center for numerical stability
        scores = scores / self.temperature
        scores = scores - scores.max(dim=-1, keepdim=True).values

        # Polynomial kernel: (1 + S/order)^order, clamped to avoid overflow
        base = 1 + scores / self.order
        # For numerical safety, avoid negative base
        base = torch.clamp(base, min=0.0)
        attn_weights = base**self.order

        # Normalize to sum = 1 (like softmax)
        attn_weights = attn_weights / (attn_weights.sum(dim=-1, keepdim=True) + 1e-8)

        # Attention mask handling (compatible with DistilBERT's additive mask)
        if attention_mask is not None:
            # Convert additive mask to multiplicative
            if attention_mask.dim() == 2:
                attn_weights = attn_weights * attention_mask[:, None, None, :].to(
                    attn_weights.dtype
                )
            elif attention_mask.dim() == 3:
                attn_weights = attn_weights * attention_mask[:, None, :, :].to(
                    attn_weights.dtype
                )
            elif attention_mask.dim() == 4:
                if attention_mask.shape[-2] == attention_mask.shape[-1]:
                    # Additive mask: 0 = attend, large negative = mask
                    attn_weights = attn_weights * (
                        attention_mask[:, 0, 0, :] > -10000.0
                    ).unsqueeze(1).unsqueeze(2).to(attn_weights.dtype)
                else:
                    attn_weights = attn_weights * attention_mask.to(attn_weights.dtype)
            attn_weights = attn_weights / (
                attn_weights.sum(dim=-1, keepdim=True) + 1e-8
            )

        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)  # (B, H, N, D)
        context = (
            context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        )
        output = self.out_proj(context)
        return (output, attn_weights) if output_attentions else (output,)

    def get_ortho_loss(self) -> Any:
        """Get ortho loss."""
        return torch.tensor(0.0, device=self.out_proj.weight.device)
