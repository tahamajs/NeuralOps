import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("moba")
@ATTENTION_REGISTRY.register("moba_attention")
class MoBAAttention(BaseAttention):
    r"""
    MoBA: Mixture of Block Attention for Long-Context Modeling.

    Mathematical Mechanism:
        Partitions the sequence into blocks of size $B_{\text{sz}}$ and routes each query block
        to top-$k$ key-value blocks based on centroid cosine affinity:
        $$\begin{aligned}
        \bar{Q}_i &= \frac{1}{B_{\text{sz}}} \sum_{t \in \text{block}_i} Q_t, \quad \bar{K}_j = \frac{1}{B_{\text{sz}}} \sum_{t \in \text{block}_j} K_t \\
        S_{\text{block}}(i, j) &= \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}} \in \mathbb{R}^{B \times H \times N_b \times N_b} \\
        \mathcal{I}_i &= \text{topk}\left(S_{\text{block}}(i, :), k\right) \\
        K_{\text{sel}, i} &= \text{gather}\left(K, \mathcal{I}_i\right) \in \mathbb{R}^{B \times H \times (k \cdot B_{\text{sz}}) \times d_k} \\
        V_{\text{sel}, i} &= \text{gather}\left(V, \mathcal{I}_i\right) \in \mathbb{R}^{B \times H \times (k \cdot B_{\text{sz}}) \times d_k} \\
        O_i &= \text{softmax}\left(\frac{Q_i K_{\text{sel}, i}^\top}{\sqrt{d_k}}\right) V_{\text{sel}, i}
        \end{aligned}$$

    Computational Complexity:
        - Block Centroid Routing: $\mathcal{O}\left(B \cdot H \cdot \frac{L^2}{B_{\text{sz}}^2} \cdot d_k\right)$
        - Selected Attention: $\mathcal{O}(B \cdot H \cdot L \cdot k \cdot B_{\text{sz}} \cdot d_k)$ — linear $\mathcal{O}(L)$ in sequence length when $k B_{\text{sz}} \ll L$.

    Tensor Dimensions:
        - Input `hidden_states`: `[B, L, D]`
        - Output: Tuple `(output, None)` where `output` is `[B, L, D]`.
    """

    def __init__(
        self,
        config: Any,
        block_size: int = 64,
        top_k: int = 4,
        rank: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads if self.num_heads > 0 else 64
        self.block_size: int = block_size
        self.top_k: int = top_k
        dropout_p: float = getattr(config, "dropout", 0.1)

        self.q_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.k_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.v_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.out_proj: nn.Linear = nn.Linear(hidden_size, hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def copy_weights(self, pretrained_attn: nn.Module) -> None:
        with torch.no_grad():
            q_src = getattr(pretrained_attn, "q_lin", getattr(pretrained_attn, "query", getattr(pretrained_attn, "q_proj", None)))
            k_src = getattr(pretrained_attn, "k_lin", getattr(pretrained_attn, "key", getattr(pretrained_attn, "k_proj", None)))
            v_src = getattr(pretrained_attn, "v_lin", getattr(pretrained_attn, "value", getattr(pretrained_attn, "v_proj", None)))
            out_src = getattr(pretrained_attn, "out_lin", getattr(pretrained_attn, "dense", getattr(pretrained_attn, "out_proj", getattr(pretrained_attn, "o_proj", None))))

            if q_src is not None and hasattr(q_src, "weight"):
                self.q_proj.weight.copy_(q_src.weight)
                if q_src.bias is not None:
                    self.q_proj.bias.copy_(q_src.bias)
            if k_src is not None and hasattr(k_src, "weight"):
                self.k_proj.weight.copy_(k_src.weight)
                if k_src.bias is not None:
                    self.k_proj.bias.copy_(k_src.bias)
            if v_src is not None and hasattr(v_src, "weight"):
                self.v_proj.weight.copy_(v_src.weight)
                if v_src.bias is not None:
                    self.v_proj.bias.copy_(v_src.bias)
            if out_src is not None and hasattr(out_src, "weight"):
                self.out_proj.weight.copy_(out_src.weight)
                if out_src.bias is not None:
                    self.out_proj.bias.copy_(out_src.bias)

def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation for MoBAAttention.

        Args:
            hidden_states: Input tensor of shape `[B, L, D]`.
            attention_mask: Optional mask tensor.
            head_mask: Optional head masking tensor.
            output_attentions: Unused flag for interface compliance.
            **kwargs: Extra keyword arguments.

        Returns:
            Tuple `(output, None)` with output of shape `[B, L, D]`.
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q = self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        pad = (self.block_size - L % self.block_size) % self.block_size
        if pad > 0:
            Q = F.pad(Q, (0, 0, 0, pad))
            K = F.pad(K, (0, 0, 0, pad))
            V = F.pad(V, (0, 0, 0, pad))
        Lp = L + pad
        nb = Lp // self.block_size

        Qb = Q.view(B, H, nb, self.block_size, d)
        Kb = K.view(B, H, nb, self.block_size, d)
        Vb = V.view(B, H, nb, self.block_size, d)

        # Block-level centroid scores
        Q_mean = Qb.mean(dim=3)
        K_mean = Kb.mean(dim=3)
        scale = 1.0 / math.sqrt(d)
        block_scores = torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale  # (B, H, nb, nb)

        is_causal = bool(kwargs.get("is_causal", getattr(self.config, "is_decoder", False) or getattr(self.config, "causal", False)))
        if is_causal:
            causal_block_mask = torch.triu(
                torch.full((nb, nb), float("-inf"), device=block_scores.device, dtype=block_scores.dtype),
                diagonal=1,
            )
            block_scores = block_scores + causal_block_mask.unsqueeze(0).unsqueeze(0)

        top_k = min(self.top_k, nb)
        _, top_idx = block_scores.topk(top_k, dim=-1)  # (B, H, nb, top_k)

        out_blocks = torch.zeros_like(Qb)
        for i in range(nb):
            idx = top_idx[:, :, i, :]  # (B, H, top_k)
            idx_exp = idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, self.block_size, d)
            K_sel = torch.gather(Kb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
            V_sel = torch.gather(Vb, 2, idx_exp).view(B, H, top_k * self.block_size, d)
            Q_i = Qb[:, :, i]  # (B, H, block_size, d)
            scores = torch.matmul(Q_i, K_sel.transpose(-2, -1)) * scale  # (B, H, block_size, top_k * block_size)
            if is_causal:
                # Key global token offsets: idx has shape (B, H, top_k)
                # Query global token offsets for block i: [i * block_size, ..., (i+1) * block_size - 1]
                q_pos = (i * self.block_size + torch.arange(self.block_size, device=scores.device)).view(1, 1, self.block_size, 1)
                # k_pos has shape (B, H, 1, top_k * block_size)
                k_block_offsets = idx.unsqueeze(-1) * self.block_size + torch.arange(self.block_size, device=scores.device).view(1, 1, 1, self.block_size)
                k_pos = k_block_offsets.view(B, H, 1, top_k * self.block_size)
                invalid = k_pos > q_pos
                scores = scores.masked_fill(invalid, float("-inf"))
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            ctx = torch.matmul(attn, V_sel)
            out_blocks[:, :, i] = ctx

        out = out_blocks.view(B, H, Lp, d)[:, :, :L, :]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None


@ATTENTION_REGISTRY.register("moba")
@ATTENTION_REGISTRY.register("moba_inject")
def inject_moba(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module] = None,
) -> MoBAAttention:
    r"""
    Injection hook for MoBAAttention into transformer layer.

    Args:
        layer: Target transformer layer module.
        model_config: Model configuration.
        comp_config: Compression configuration.
        layer_idx: Layer index.
        total_layers: Total number of transformer layers.
        old_attn: Previous attention module.

    Returns:
        Instantiated `MoBAAttention` assigned to `layer.attention`.
    """
    block_size: int = getattr(comp_config, "block_size", 64)
    top_k: int = getattr(comp_config, "top_k", 4)
    new_attn = MoBAAttention(
        config=model_config,
        block_size=block_size,
        top_k=top_k,
    )
    layer.attention = new_attn
    return new_attn
