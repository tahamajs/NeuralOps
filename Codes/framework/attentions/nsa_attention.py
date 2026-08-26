r"""
Native Sparse Attention (NSA) Module.

Combines sliding-window local context capture with dynamic, input-dependent top-$k$
global routing across key-value representations in a fully vectorized manner.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nsa")
@ATTENTION_REGISTRY.register("nsa_attention")
class NativeSparseAttention(BaseAttention):
    r"""
    Native Sparse Attention layer combining local windowing and global top-$k$ selection.

    Mathematical Formulation:
        1. Dense Query-Key Similarity:
           $$S_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}}$$
        
        2. Vectorized Hybrid Masking:
           $$\mathcal{M}^{\text{local}}_{ij} = \mathbb{I}(|i - j| \le W)$$
           $$\mathcal{M}^{\text{topk}}_{ij} = \mathbb{I}(j \in \operatorname{top-}k(\{ S_{ij'} : |i - j'| > W \}))$$
           $$\mathcal{M}_{ij} = \mathcal{M}^{\text{local}}_{ij} \lor \mathcal{M}^{\text{topk}}_{ij}$$
        
        3. Sparse Context Evaluation:
           $$A = \operatorname{Softmax}(S + (1 - \mathcal{M}) \cdot (-\infty) + M_{\text{pad}})$$
           $$O = A V W_O$$

    Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot L \cdot (2W + k) \cdot d)$
        - Memory: $\mathcal{O}(B \cdot H \cdot L^2)$
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        window_size: int = 32,
        top_k: int = 16,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 4)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.window_size: int = getattr(config, "nsa_window_size", window_size)
        self.top_k: int = getattr(config, "nsa_top_k", top_k)

        self.q_proj = nn.Linear(hidden_size, self.all_head_size)
        self.k_proj = nn.Linear(hidden_size, self.all_head_size)
        self.v_proj = nn.Linear(hidden_size, self.all_head_size)
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
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        scale = d ** -0.5
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B, H, L, L]

        # Valid mask handling
        valid_mask_2d = get_token_valid_mask(attention_mask, B, L, device)  # [B, L]
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)              # [B, 1, 1, L]

        # Local sliding window boolean mask
        idx = torch.arange(L, device=device)
        local_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size  # [L, L]
        local_mask_4d = local_mask.unsqueeze(0).unsqueeze(0)                          # [1, 1, L, L]

        if L <= (2 * self.window_size + 1):
            # Window covers full sequence -> exact attention on valid tokens
            total_mask = local_mask_4d & key_valid_4d
        else:
            # Vectorized top-k selection for non-local tokens
            non_local_scores = scores.masked_fill(local_mask_4d | (~key_valid_4d), -10000.0)
            k_val = min(self.top_k, max(1, L - (2 * self.window_size + 1)))
            _, topk_ind = torch.topk(non_local_scores, k=k_val, dim=-1)  # [B, H, L, k_val]
            topk_mask = torch.zeros_like(scores, dtype=torch.bool).scatter_(-1, topk_ind, True)
            total_mask = (local_mask_4d | topk_mask) & key_valid_4d

        scores = scores.masked_fill(~total_mask, -10000.0)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        if head_mask is not None:
            attn = attn * head_mask

        context = torch.matmul(attn, V)
        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        output = self.out_proj(context)
        return output, (attn if output_attentions else None)


@ATTENTION_REGISTRY.register("nsa_inject")
def inject_nsa(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
    **kwargs) -> NativeSparseAttention:
    window_size = getattr(comp_config, "window_size", 32)
    top_k = getattr(comp_config, "top_k", 16)
    new_attn = NativeSparseAttention(
        config=model_config,
        window_size=window_size,
        top_k=top_k,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
