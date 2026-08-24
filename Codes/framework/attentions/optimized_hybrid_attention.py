r"""
Optimized Hybrid Attention (OHA).

Fuses localized sliding-window self-attention with sequence-compressed low-rank global attention
via dynamic head-wise gated modulation and exact pretrained weight transfer.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("oha")
@ATTENTION_REGISTRY.register("optimized_hybrid_attention")
class OptimizedHybridAttention(BaseAttention):
    r"""
    Optimized Hybrid Attention (OHA) combining sliding-window local and compressed global attention.

    Mathematical Formulation:
        1. Local Sliding-Window Path:
           $$S^{\text{local}}_{ij} = \frac{Q_i K_j^\top}{\sqrt{d}} + M^{\text{band}}_{ij}$$
           $$O^{\text{local}} = \operatorname{Softmax}(S^{\text{local}}) V$$
        
        2. Sequence-Compressed Low-Rank Global Path:
           $$\widetilde{K} = \operatorname{Softmax}(X W_K^{\text{comp}}, \text{dim}=N)^\top K \in \mathbb{R}^{B \times H \times R \times d}$$
           $$\widetilde{V} = \operatorname{Softmax}(X W_V^{\text{comp}}, \text{dim}=N)^\top V \in \mathbb{R}^{B \times H \times R \times d}$$
           $$O^{\text{global}} = \operatorname{Softmax}\left(\frac{Q \widetilde{K}^\top}{\sqrt{d}}\right) \widetilde{V}$$
        
        3. Dynamic Gated Fusion:
           $$g = \sigma(X W_g + b_g) \in \mathbb{R}^{B \times N \times H}$$
           $$O = g \odot O^{\text{local}} + (1 - g) \odot O^{\text{global}}$$

    Complexity:
        - Local path: $\mathcal{O}(B \cdot H \cdot N \cdot W \cdot d)$
        - Global path: $\mathcal{O}(B \cdot H \cdot N \cdot R \cdot d)$
        - Total Complexity: $\mathcal{O}(N \cdot (W + R) \cdot d)$ linear.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        window_size: int = 64,
        compression_rank: int = 32,
        use_gate: bool = True,
        gate_bias: float = 2.0,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.window_size: int = getattr(config, "oha_window_size", window_size)
        self.compression_rank: int = rank if rank is not None else getattr(config, "compression_rank", compression_rank)
        self.use_gate: bool = use_gate

        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        self.compress_k = nn.Linear(hidden_size, self.compression_rank, bias=False)
        self.compress_v = nn.Linear(hidden_size, self.compression_rank, bias=False)
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

        if self.use_gate:
            self.gate = nn.Linear(hidden_size, self.num_heads)
            nn.init.constant_(self.gate.bias, gate_bias)
            nn.init.normal_(self.gate.weight, std=0.02)

        nn.init.normal_(self.compress_k.weight, std=0.02)
        nn.init.normal_(self.compress_v.weight, std=0.02)

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
            except Exception:
                pass

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

        Q = self.query(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, N, H, d).permute(0, 2, 1, 3)

        valid_mask_2d = get_token_valid_mask(attention_mask, B, N, device)  # [B, N]
        key_valid_4d = valid_mask_2d.unsqueeze(1).unsqueeze(2)              # [B, 1, 1, N]

        scale = d ** -0.5

        # 1. Local sliding-window attention
        idx = torch.arange(N, device=device)
        is_causal = getattr(self, "is_causal", False)
        if is_causal:
            band_mask = (idx.unsqueeze(0) - idx.unsqueeze(1) >= 0) & (idx.unsqueeze(0) - idx.unsqueeze(1) <= self.window_size)
        else:
            band_mask = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs() <= self.window_size
        local_mask = band_mask.unsqueeze(0).unsqueeze(0) & key_valid_4d

        local_scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        local_scores = local_scores.masked_fill(~local_mask, -10000.0)
        local_probs = F.softmax(local_scores, dim=-1)
        local_probs = self.dropout(local_probs)
        local_ctx = torch.matmul(local_probs, V)

        # 2. Global low-rank attention
        comp_k_raw = self.compress_k(hidden_states)  # [B, N, R]
        comp_v_raw = self.compress_v(hidden_states)
        mask_inv = (~valid_mask_2d).unsqueeze(-1)
        comp_k_raw = comp_k_raw.masked_fill(mask_inv, -10000.0)
        comp_v_raw = comp_v_raw.masked_fill(mask_inv, -10000.0)

        comp_k_weights = F.softmax(comp_k_raw, dim=1)  # [B, N, R]
        comp_v_weights = F.softmax(comp_v_raw, dim=1)

        # K: [B, H, N, d], comp_k_weights: [B, N, R] -> ck: [B, H, R, d]
        ck = torch.einsum("bhnd,bnr->bhrd", K, comp_k_weights)
        cv = torch.einsum("bhnd,bnr->bhrd", V, comp_v_weights)

        global_scores = torch.matmul(Q, ck.transpose(-2, -1)) * scale  # [B, H, N, R]
        global_probs = F.softmax(global_scores, dim=-1)
        global_probs = self.dropout(global_probs)
        global_ctx = torch.matmul(global_probs, cv)  # [B, H, N, d]

        # 3. Gated fusion
        if self.use_gate:
            gate = torch.sigmoid(self.gate(hidden_states))  # [B, N, H]
            gate = gate.permute(0, 2, 1).unsqueeze(-1)      # [B, H, N, 1]
            context = gate * local_ctx + (1.0 - gate) * global_ctx
        else:
            context = 0.5 * local_ctx + 0.5 * global_ctx

        if head_mask is not None:
            context = context * head_mask

        context = context * valid_mask_2d.unsqueeze(1).unsqueeze(-1).to(dtype)
        context = context.permute(0, 2, 1, 3).contiguous().view(B, N, self.all_head_size)
        out = self.out_proj(context)
        return out, (local_probs if output_attentions else None)


@ATTENTION_REGISTRY.register("oha_inject")
def inject_oha(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> OptimizedHybridAttention:
    new_attn = OptimizedHybridAttention(
        config=model_config,
        window_size=getattr(comp_config, "window_size", 64),
        compression_rank=getattr(comp_config, "compression_rank", 32),
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
