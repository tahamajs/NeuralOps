"""
AdaSplash Attention Module with α-Entmax Sparse Activations.
Fixes:
- Correct entmax15 bisection algorithm.
- Proper mask conversion to additive.
- Added injection function with pretrained weight copy.
- Support head_mask.
"""

from typing import Any, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


def entmax15_bisect(scores: torch.Tensor, dim: int = -1, n_iter: int = 50) -> torch.Tensor:
    """Compute 1.5-entmax via bisection, ensuring sums to 1.
    Based on the sparsegen implementation.
    """
    alpha = 1.5
    alpha_minus_1 = alpha - 1.0

    # Shift scores for numerical stability
    max_scores = scores.max(dim=dim, keepdim=True).values
    scores_shift = scores - max_scores
    scaled = scores_shift * alpha_minus_1

    # Initial interval for tau
    tau_lo = scaled.min(dim=dim, keepdim=True).values - 1.0
    tau_hi = scaled.max(dim=dim, keepdim=True).values

    for _ in range(n_iter):
        tau = (tau_lo + tau_hi) / 2.0
        p = torch.clamp(scaled - tau, min=0.0) ** (1.0 / alpha_minus_1)
        sum_p = p.sum(dim=dim, keepdim=True)
        # If sum > 1, tau is too low -> increase tau_lo
        mask = sum_p > 1.0
        tau_lo = torch.where(mask, tau, tau_lo)
        tau_hi = torch.where(mask, tau_hi, tau)

    # Final tau
    tau = (tau_lo + tau_hi) / 2.0
    p = torch.clamp(scaled - tau, min=0.0) ** (1.0 / alpha_minus_1)
    return p


@ATTENTION_REGISTRY.register("adasplash")
class AdaSplashAttention(BaseAttention):
    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        alpha: float = 1.5,
        n_iter: int = 10,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config, rank)
        self.alpha = alpha
        self.n_iter = n_iter
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads

        self.q_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size)
        dropout_p = getattr(config, "dropout", 0.1)
        self.dropout = nn.Dropout(dropout_p)

        # Copy pretrained weights if available
        if pretrained_attn is not None:
            with torch.no_grad():
                self.q_proj.weight.data.copy_(pretrained_attn.q_lin.weight.data)
                self.q_proj.bias.data.copy_(pretrained_attn.q_lin.bias.data)
                self.k_proj.weight.data.copy_(pretrained_attn.k_lin.weight.data)
                self.k_proj.bias.data.copy_(pretrained_attn.k_lin.bias.data)
                self.v_proj.weight.data.copy_(pretrained_attn.v_lin.weight.data)
                self.v_proj.bias.data.copy_(pretrained_attn.v_lin.bias.data)
                self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

    def _convert_mask_to_additive(self, attention_mask, B, N, device, dtype):
        """Return additive mask [B,1,1,N] with 0 for valid, -10000 for padding."""
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
                # Assume additive (0 and -inf/-10000); already in correct form
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
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q = self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)

        logits = torch.matmul(q, k.transpose(-2, -1)) / (Dh ** 0.5)

        # Apply mask
        additive_mask = self._convert_mask_to_additive(attention_mask, B, L, q.device, logits.dtype)
        if additive_mask is not None:
            logits = logits + additive_mask

        # Apply entmax
        if self.alpha == 1.0:
            attn = F.softmax(logits, dim=-1)
        elif self.alpha == 1.5:
            attn = entmax15_bisect(logits, dim=-1, n_iter=self.n_iter)
        else:
            raise ValueError("Only alpha=1.0 or 1.5 supported in this version.")

        attn = self.dropout(attn)
        if head_mask is not None:
            attn = attn * head_mask

        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, L, D)
        return self.out_proj(out), (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adasplash_inject")
def inject_adasplash(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AdaSplashAttention:
    alpha = getattr(comp_config, "alpha", 1.5)
    new_attn = AdaSplashAttention(
        model_config,
        alpha=alpha,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn