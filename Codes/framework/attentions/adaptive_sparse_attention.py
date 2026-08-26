"""
Adaptive Sparse Attention Module.
Implements dynamic top-k and threshold-based adaptive sparse attention.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("adaptive_sparse_attention")
@ATTENTION_REGISTRY.register("adaptive_sparse")
class AdaptiveSparseAttention(BaseAttention):
    r"""
    Adaptive Sparse Multi-Head Attention.
    
    Mathematical Formulation:
        1. Query-Key Attention Scores:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$
        
        2. Adaptive Dynamic Sparsification:
           Per-head learnable sparsity ratio $\gamma_h = \sigma(\text{logit}_h)$.
           Top-$k$ active key positions are retained per query:
           $$k_h = \max\left(1, \lfloor N \cdot (1 - \gamma_h) \rfloor\right)$$
           $$\mathcal{M}_{ij}^{(h)} = \begin{cases} 0 & \text{if } S_{ij}^{(h)} \ge \text{TopK-Threshold}_h \\ -\infty & \text{otherwise} \end{cases}$$
        
        3. Context Aggregation:
           $$A = \operatorname{Softmax}(S + \mathcal{M} + M_{\text{pad}})$$
           $$O = (A V) W_O$$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        sparsity_ratio: float = 0.5,
        dropout: float = 0.1,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.default_sparsity: float = float(sparsity_ratio)

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
        self.out_proj = nn.Linear(self.all_head_size, hidden_size)

        # Learnable per-head sparsity logit initialized to target sparsity
        init_logit = math.log(max(1e-4, min(0.9999, sparsity_ratio)) / (1.0 - max(1e-4, min(0.9999, sparsity_ratio))))
        self.sparsity_logits = nn.Parameter(torch.full((self.num_heads,), fill_value=init_logit))

        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", dropout),
            ),
        )
        self.dropout = nn.Dropout(dropout_p)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Universal adapter weight copying."""
        from framework.core.adapters import UniversalAdapter

        for proj, target in [("q", self.q_proj), ("k", self.k_proj), ("v", self.v_proj), ("o", self.out_proj)]:
            try:
                w, b = UniversalAdapter.get_weight_bias(src, proj)
                target.weight.data.copy_(w)
                if b is not None and target.bias is not None:
                    target.bias.data.copy_(b)
            except (AttributeError, KeyError, RuntimeError, TypeError, ValueError):
                pass

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

        Q = self.q_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        K = self.k_proj(hidden_states).view(B, N, H, d).transpose(1, 2)
        V = self.v_proj(hidden_states).view(B, N, H, d).transpose(1, 2)

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, N, N]

        # Valid padding mask
        valid_mask = get_token_valid_mask(attention_mask, B, N, device)  # [B, N]
        if valid_mask is not None:
            pad_mask = (~valid_mask).unsqueeze(1).unsqueeze(2)  # [B, 1, 1, N]
            scores = scores.masked_fill(pad_mask, -10000.0)

        # Dynamic Top-K Adaptive Sparsification per head
        sparsity_per_head = torch.sigmoid(self.sparsity_logits).clamp(0.0, 0.95)  # [H]
        k_per_head = torch.clamp(
            ((1.0 - sparsity_per_head) * N).long(),
            min=1,
            max=N,
        )  # [H]

        # Apply top-k mask per head
        sparse_scores = []
        for h in range(H):
            k_val = int(k_per_head[h].item())
            h_scores = scores[:, h, :, :]  # [B, N, N]
            if k_val < N:
                topk_thresh = torch.topk(h_scores, k=k_val, dim=-1, largest=True).values[..., -1:]
                h_scores = h_scores.masked_fill(h_scores < topk_thresh, -10000.0)
            sparse_scores.append(h_scores.unsqueeze(1))

        scores = torch.cat(sparse_scores, dim=1)  # [B, H, N, N]

        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        if valid_mask is not None:
            context = context * valid_mask.unsqueeze(1).unsqueeze(-1).to(dtype)

        context = context.transpose(1, 2).contiguous().view(B, N, self.all_head_size)
        output = self.out_proj(context)

        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("adaptive_sparse_attention_inject")
@ATTENTION_REGISTRY.register("adaptive_sparse_inject")
def inject_adaptive_sparse(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> AdaptiveSparseAttention:
    sparsity_ratio = getattr(comp_config, "adaptive_sparsity_ratio", 0.5)
    dropout = getattr(comp_config, "dropout", 0.1)
    new_attn = AdaptiveSparseAttention(
        config=model_config,
        sparsity_ratio=sparsity_ratio,
        dropout=dropout,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
