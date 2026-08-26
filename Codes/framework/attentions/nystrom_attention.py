r"""
Nyströmformer Attention – Nyström-based low-rank kernel approximation.
Paper: https://arxiv.org/abs/2102.03902
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nystrom")
@ATTENTION_REGISTRY.register("nystrom_attention")
class NystromAttention(BaseAttention):
    r"""
    Nyströmformer: Linear Complexity Self-Attention via Nyström Landmark Approximation (Xiong et al., 2021).

    Mathematical Formulation:
        $$\hat{A} = \hat{F}_1 \hat{F}_2^+ \hat{F}_3$$
        $$\hat{F}_1 = \operatorname{Softmax}\left(\frac{Q \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{N \times m}, \quad \hat{F}_2 = \operatorname{Softmax}\left(\frac{\tilde{Q} \tilde{K}^\top}{\sqrt{d_k}}\right) \in \mathbb{R}^{m \times m}, \quad \hat{F}_3 = \operatorname{Softmax}\left(\frac{\tilde{Q} K^\top}{\sqrt{d_k}} + M\right) \in \mathbb{R}^{m \times N}$$
        $$\text{Attention}_{\text{Nyström}}(Q, K, V) = \hat{F}_1 \left( \hat{F}_2^+ \left( \hat{F}_3 V \right) \right)$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot (N \cdot m \cdot d_k + m^3))$
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot m + m \cdot d_k))$ vs $\mathcal{O}(B \cdot H \cdot N^2)$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        num_landmarks: Optional[int] = None,
        pinv_iterations: int = 6,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.num_landmarks: int = (
            num_landmarks
            if num_landmarks is not None
            else getattr(config, "num_landmarks", getattr(config, "nystrom_landmarks", rank if rank is not None else 64))
        )
        self.pinv_iterations: int = pinv_iterations

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

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

    def _iterative_pinv(self, A: torch.Tensor, iters: int = 6) -> torch.Tensor:
        r"""Compute Moore-Penrose pseudo-inverse using iterative Schulz updates."""
        norm_1 = A.abs().sum(dim=-2).amax(dim=-1, keepdim=True).unsqueeze(-1)
        norm_inf = A.abs().sum(dim=-1).amax(dim=-1, keepdim=True).unsqueeze(-1)
        alpha = 1.0 / (norm_1 * norm_inf + 1e-5)

        V = alpha * A.transpose(-2, -1)
        I = torch.eye(A.size(-1), device=A.device, dtype=A.dtype).unsqueeze(0).unsqueeze(0)

        for _ in range(iters):
            V = torch.matmul(V, 2.0 * I - torch.matmul(A, V))
        return V

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

        Q = self.query(hidden_states).view(B, N, H, d).transpose(1, 2)
        K = self.key(hidden_states).view(B, N, H, d).transpose(1, 2)
        V = self.value(hidden_states).view(B, N, H, d).transpose(1, 2)

        m = min(self.num_landmarks, N)

        # Adaptive pooling landmarks
        Q_flat = Q.reshape(B * H, N, d).transpose(1, 2)
        K_flat = K.reshape(B * H, N, d).transpose(1, 2)

        Q_landmarks = F.adaptive_avg_pool1d(Q_flat, m).transpose(1, 2).reshape(B, H, m, d)
        K_landmarks = F.adaptive_avg_pool1d(K_flat, m).transpose(1, 2).reshape(B, H, m, d)

        scale = d ** -0.5
        F1 = F.softmax(torch.matmul(Q, K_landmarks.transpose(-2, -1)) * scale, dim=-1)
        F2 = F.softmax(torch.matmul(Q_landmarks, K_landmarks.transpose(-2, -1)) * scale, dim=-1)

        scores3 = torch.matmul(Q_landmarks, K.transpose(-2, -1)) * scale
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # (B, N)
        mask_expanded = valid_mask.unsqueeze(1).unsqueeze(2)             # (B, 1, 1, N)
        scores3 = scores3.masked_fill(~mask_expanded, -10000.0)
        F3 = F.softmax(scores3, dim=-1)

        F2_pinv = self._iterative_pinv(F2, iters=self.pinv_iterations)

        # Associative linear multiplication: O(N * m * d)
        step1 = torch.matmul(F3, V)
        step2 = torch.matmul(F2_pinv, step1)
        context = torch.matmul(F1, step2)

        context = self.dropout(context)
        if head_mask is not None:
            context = context * head_mask

        context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, None


@ATTENTION_REGISTRY.register("nystrom_inject")
def inject_nystrom(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
,
    **kwargs) -> NystromAttention:
    landmarks = getattr(comp_config, "num_landmarks", 64)
    new_attn = NystromAttention(
        config=model_config,
        num_landmarks=landmarks,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
