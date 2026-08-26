r"""
Performer Attention – Fast Attention via Positive Orthogonal Random Features (FAVOR+).
Paper: https://arxiv.org/abs/2009.14794
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
@ATTENTION_REGISTRY.register("performer_attention")
class PerformerAttention(BaseAttention):
    r"""
    Performer Attention (FAVOR+).

    Mathematical Formulation:
        1. Positive Orthogonal Random Features:
           $$\phi(x) = \frac{\exp\left(x W^\top - \frac{\|x\|^2}{2}\right)}{\sqrt{m}}$$
           where $W \in \mathbb{R}^{m \times d}$ is an orthogonal random projection matrix ($W W^\top \approx I$).
        
        2. Linear Attention Kernel Formulation:
           $$\operatorname{Attn}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1}_N) + \epsilon}$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot m \cdot d)$ linear.
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot m + m \cdot d))$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        eps: float = 1e-6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.num_features: int = getattr(
            config,
            "performer_rank",
            rank if rank is not None else getattr(config, "rank", 64),
        )
        self.kernel_type: str = getattr(config, "performer_kernel_type", kwargs.get("kernel_type", "favor_plus"))
        self.eps: float = eps

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        omega = self._build_orthogonal_matrix(self.head_dim, self.num_features)
        self.register_buffer("omega", omega)

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        r"""Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.query), ("k", self.key), ("v", self.value), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

    def _build_orthogonal_matrix(self, d: int, m: int) -> torch.Tensor:
        r"""Constructs (m, d) matrix with orthogonal rows scaled by sqrt(d)."""
        blocks = []
        remaining = m
        while remaining > 0:
            block_size = min(d, remaining)
            g = torch.randn(d, block_size)
            q, _ = torch.linalg.qr(g)
            blocks.append(q)
            remaining -= block_size
        W = torch.cat(blocks, dim=1)
        W = W * math.sqrt(d)
        return W.t().contiguous()

    def _feature_map(self, x: torch.Tensor) -> torch.Tensor:
        r"""x: (B, H, N, d) -> phi: (B, H, N, m)
        FAVOR+ Positive Orthogonal Random Feature Map.
        Approximates the softmax Gaussian kernel via:
        phi(x) = exp(x W^T - ||x||^2 / 2) / sqrt(m)
        with log-sum-exp shift for numerical stability.
        """
        proj = torch.matmul(x, self.omega.to(dtype=x.dtype, device=x.device).t())
        if self.kernel_type in ["favor_plus", "exp", "favor"]:
            diag_sq = torch.sum(x ** 2, dim=-1, keepdim=True) * 0.5  # (B, H, N, 1)
            max_proj = proj.amax(dim=-1, keepdim=True).detach()
            phi = torch.exp(proj - max_proj - diag_sq) + self.eps
            phi = phi / (math.sqrt(self.num_features) + self.eps)
        elif self.kernel_type == "elu":
            phi = (F.elu(proj) + 1.0) / (math.sqrt(self.num_features) + self.eps)
        else:
            phi = (F.relu(proj) + self.eps) / (math.sqrt(self.num_features) + self.eps)
        return phi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Any] = None,
        output_attentions: Optional[bool] = False,
        output_hidden_states: Optional[bool] = False,
        return_dict: Optional[bool] = True,
        *args: Any,
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

        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)   # (B, 1, N, 1)

        Q_feat = self._feature_map(Q)
        K_feat = self._feature_map(K) * mask_expanded

        # Associative linear attention reduction
        KV = torch.matmul(K_feat.transpose(-2, -1), V)   # (B, H, m, d)
        K_sum = K_feat.sum(dim=-2, keepdim=True)         # (B, H, 1, m)
        num = torch.matmul(Q_feat, KV)                   # (B, H, N, d)
        den = torch.matmul(Q_feat, K_sum.transpose(-2, -1)) + self.eps
        context = num / den

        context = self.dropout(context)
        if head_mask is not None:
            context = context * head_mask

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
,
    **kwargs) -> PerformerAttention:
    rank = getattr(comp_config, "performer_rank", 64)
    new_attn = PerformerAttention(
        config=model_config,
        rank=rank,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn