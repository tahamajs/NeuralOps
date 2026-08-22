"""
Performer Attention – Fast Attention via Positive Orthogonal Random Features (FAVOR+).
Fixes:
- Added injection function with pretrained weight copy.
- Fixed orthogonal projection matrix generation.
- Proper mask handling for padding.
"""

import math
from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(nn.Module):
    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // self.num_heads
        self.all_head_size = self.num_heads * self.head_dim
        self.num_features = getattr(
            config,
            "performer_rank",
            rank if rank is not None else getattr(config, "rank", 64),
        )
        self.eps = eps

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)

        # Initialize from pretrained if provided
        if pretrained_attn is not None:
            with torch.no_grad():
                self.query.weight.data.copy_(pretrained_attn.q_lin.weight.data)
                self.query.bias.data.copy_(pretrained_attn.q_lin.bias.data)
                self.key.weight.data.copy_(pretrained_attn.k_lin.weight.data)
                self.key.bias.data.copy_(pretrained_attn.k_lin.bias.data)
                self.value.weight.data.copy_(pretrained_attn.v_lin.weight.data)
                self.value.bias.data.copy_(pretrained_attn.v_lin.bias.data)
                self.out_proj.weight.data.copy_(pretrained_attn.out_lin.weight.data)
                self.out_proj.bias.data.copy_(pretrained_attn.out_lin.bias.data)

        # Correct orthogonal random features
        omega = self._build_orthogonal_matrix(self.head_dim, self.num_features)
        self.register_buffer("omega", omega)   # shape (m, d)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(config, "hidden_dropout_prob",
                    getattr(config, "attention_probs_dropout_prob", 0.1)),
        )
        self.dropout = nn.Dropout(dropout_p)

    def _build_orthogonal_matrix(self, d: int, m: int) -> torch.Tensor:
        """Generate (m, d) matrix with orthogonal rows? Actually we need (m, d) for x @ omega.T -> (..., m).
        We construct a (d, m) matrix with orthonormal columns, then transpose to (m, d).
        """
        blocks = []
        remaining = m
        while remaining > 0:
            block_size = min(d, remaining)
            g = torch.randn(d, block_size)
            q, _ = torch.linalg.qr(g)   # q is (d, block_size) with orthonormal columns
            blocks.append(q)
            remaining -= block_size
        # Concatenate columns: shape (d, m)
        W = torch.cat(blocks, dim=1)
        # Scale by sqrt(d) as per FAVOR+ theory
        W = W * math.sqrt(d)
        # Return transpose: (m, d)
        return W.t().contiguous()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, H, N, d) -> phi: (B, H, N, m)"""
        proj = torch.matmul(x, self.omega.to(dtype=x.dtype, device=x.device).t())  # x @ omega.t -> (..., m)
        x_norm_sq = torch.sum(x ** 2, dim=-1, keepdim=True) / 2.0
        logits = proj - x_norm_sq
        # Stabilization
        max_val = logits.amax(dim=-1, keepdim=True).detach()
        phi = torch.exp(logits - max_val) / math.sqrt(self.num_features)
        return phi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, N, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        scale = d ** -0.25
        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2) * scale
        K = self.key(hidden_states).view(B, N, H, d).transpose(1, 2) * scale
        V = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        # Get valid mask: True for valid tokens
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N) bool
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)   # (B,1,N,1)

        Q_feat = self._feature_map(Q)
        K_feat = self._feature_map(K) * mask_expanded

        # Linear attention
        KV = torch.matmul(K_feat.transpose(-2, -1), V)   # (B,H,m,d)
        K_sum = K_feat.sum(dim=-2, keepdim=True)         # (B,H,1,m)
        num = torch.matmul(Q_feat, KV)                   # (B,H,N,d)
        den = torch.matmul(Q_feat, K_sum.transpose(-2, -1)) + self.eps
        context = num / den

        context = self.dropout(context)
        if head_mask is not None:
            context = context * head_mask

        # Mask padding queries
        context = context * mask_expanded
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)

        return out, None


@ATTENTION_REGISTRY.register("performer_inject")
def inject_performer(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> PerformerAttention:
    rank = getattr(comp_config, "performer_rank", 64)
    new_attn = PerformerAttention(
        model_config,
        rank=rank,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn