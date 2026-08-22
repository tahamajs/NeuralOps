"""
Adaptive Span Attention – corrected version.
Fixes:
- Initialize span_logits to zero (avoid saturation).
- Unified mask handling (bool/0-1/additive).
- Apply head_mask.
- Add class registration and separate injection.
"""
from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("adaptive_span")
class AdaptiveSpanAttention(nn.Module):
    def __init__(
        self,
        model_config: Any,
        rank: int = 32,
        dropout: float = 0.1,
        max_span: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if pretrained_attn is None:
            raise ValueError("pretrained_attn required for zero-shot initialization")

        self.num_heads = model_config.num_attention_heads
        self.head_dim = model_config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.hidden_size = model_config.hidden_size
        self.max_span = max_span

        self.q_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(self.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, self.hidden_size)

        with torch.no_grad():
            self.q_proj.weight.data.copy_(pretrained_attn.q_lin.weight.data)
            self.q_proj.bias.data.copy_(pretrained_attn.q_lin.bias.data)
            self.k_proj.weight.data.copy_(pretrained_attn.k_lin.weight.data)
            self.k_proj.bias.data.copy_(pretrained_attn.k_lin.bias.data)
            self.v_proj.weight.data.copy_(pretrained_attn.v_lin.weight.data)
            self.v_proj.bias.data.copy_(pretrained_attn.v_lin.bias.data)
            self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
            self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # Avoid saturation: initialize to 0, giving span ~ max_span/2
        self.span_logits = nn.Parameter(torch.zeros(self.num_heads))
        self.dropout = nn.Dropout(dropout)

    def _reshape(self, x: torch.Tensor, B: int) -> torch.Tensor:
        return x.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _convert_mask_to_additive(self, attention_mask, B, N, device, dtype):
        """Return additive mask with 0 for valid and -10000 for pad."""
        if attention_mask is None:
            return None
        if attention_mask.dim() == 2:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask
            else:
                valid = attention_mask > 0
            return (~valid).unsqueeze(1).unsqueeze(2).to(dtype) * -10000.0
        elif attention_mask.dim() == 4:
            if attention_mask.dtype == torch.bool:
                valid = attention_mask[:, 0, 0, :]
                return (~valid).unsqueeze(1).unsqueeze(2).to(dtype) * -10000.0
            else:
                # Assume already additive (0 and -inf/-10000)
                return attention_mask.to(dtype)
        else:
            raise ValueError(f"Unsupported attention_mask shape: {attention_mask.shape}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, _ = hidden_states.shape
        H, D = self.num_heads, self.head_dim

        Q = self._reshape(self.q_proj(hidden_states), B)
        K = self._reshape(self.k_proj(hidden_states), B)
        V = self._reshape(self.v_proj(hidden_states), B)

        scale = D ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale

        # Adaptive span mask
        span = torch.sigmoid(self.span_logits) * self.max_span   # [H]
        pos = torch.arange(N, device=Q.device).float()
        dist = (pos.unsqueeze(0) - pos.unsqueeze(1)).abs()       # [N,N]
        # span_bool: [H,N,N] True where within span
        span_bool = dist.unsqueeze(0) <= span.unsqueeze(1).unsqueeze(2)
        scores = scores.masked_fill(~span_bool.unsqueeze(0), -10000.0)  # expand to [B,H,N,N]

        # Padding mask
        additive_mask = self._convert_mask_to_additive(attention_mask, B, N, Q.device, scores.dtype)
        if additive_mask is not None:
            scores = scores + additive_mask

        attn_weights = F.softmax(scores, dim=-1)
        if head_mask is not None:
            attn_weights = attn_weights * head_mask
        attn_weights = self.dropout(attn_weights)

        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(B, N, -1)
        output = self.out_proj(context)

        return output, attn_weights if output_attentions else None


@ATTENTION_REGISTRY.register("adaptive_span_inject")
def inject_adaptive_span(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AdaptiveSpanAttention:
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