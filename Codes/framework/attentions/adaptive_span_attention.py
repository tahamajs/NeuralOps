"""
Adaptive Span Attention – Learns per‑head attention span.
Paper: "Adaptive Attention Span in Transformers" (Sukhbaatar et al.)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


class AdaptiveSpanAttention(nn.Module):
    """
    Adaptive Span Attention for DistilBERT.
    Each head learns a soft mask over the context length.
    """

    def __init__(
        self,
        model_config,
        rank: int = 32,
        dropout: float = 0.1,
        max_span: int = 128,
        pretrained_attn=None,
    ):
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero‑shot init")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.max_span = max_span

        # Projections (copied from pretrained)
        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        # Copy pretrained weights
        with torch.no_grad():
            self.q_proj.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.q_proj.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.k_proj.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.k_proj.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.v_proj.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.v_proj.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # Learnable per‑head span parameters (logits)
        self.span_logits = nn.Parameter(torch.full((self.num_heads,), float(max_span)))
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

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

        Q = self._reshape(self.q_proj(hidden_states), B)  # (B, H, N, D)
        K = self._reshape(self.k_proj(hidden_states), B)
        V = self._reshape(self.v_proj(hidden_states), B)

        # Compute raw scores
        scale = D**-0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # (B, H, N, N)

        # Adaptive span mask (soft, differentiable)
        span = torch.sigmoid(self.span_logits) * self.max_span  # (H,)
        pos = torch.arange(N, device=Q.device).float()
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()  # (N, N)
        # Span mask: 1 if within span, 0 otherwise (soft version)
        mask_span = (dist <= span.unsqueeze(1).unsqueeze(2)).float()
        # Apply mask (set scores outside span to -inf)
        scores = scores.masked_fill(mask_span == 0, -1e9)

        # Standard attention masking (padding)
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :]  # (B,1,1,N)
            else:
                mask = attention_mask
            # Convert boolean mask to float (True -> 1.0, False -> 0.0)
            mask_float = mask.float()
            scores = scores + (1.0 - mask_float) * -1e9

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


# ----------------------------------------------------------------------
# Injection function – matches the framework’s required signature
# ----------------------------------------------------------------------
@ATTENTION_REGISTRY.register("adaptive_span")
def inject_adaptive_span(
    layer, model_config, comp_config, layer_idx, total_layers, old_attn
):
    """Injection function for DistilBERT."""
    rank = getattr(comp_config, "adaptive_span_rank", 32)
    dropout = getattr(comp_config, "dropout", 0.1)
    max_span = getattr(comp_config, "max_span", 128)
    new_attn = AdaptiveSpanAttention(
        model_config=model_config,
        rank=rank,
        dropout=dropout,
        max_span=max_span,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
