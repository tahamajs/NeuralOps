r"""
XAttention: Block Sparse Attention with Coarse-to-Fine Routing.
Paper: https://arxiv.org/abs/2305.02102

Partitions sequence tokens into non-overlapping blocks, evaluates coarse block-level routing,
and computes unified normalized fine-grained attention across top-k candidate blocks.
"""

from typing import Any, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("xattention")
class XAttention(BaseAttention):
    r"""
    XAttention: Block-Sparse Attention Mechanism with Unified Probability Normalization.

    Mathematical Formulation:
        1. Block Partitioning:
           Sequence $N$ is divided into $M = \lceil N / B_s \rceil$ blocks of size $B_s$.
        
        2. Coarse Block Summaries & Routing:
           $\bar{Q}_i = \frac{1}{B_s} \sum_{t \in \text{Blk}_i} Q_t, \quad \bar{K}_j = \frac{1}{B_s} \sum_{t \in \text{Blk}_j} K_t$
           $S_{ij}^{\text{block}} = \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}}$
           $\text{TopK Blocks} = \operatorname{TopK}(S_{i, :}^{\text{block}}, k)$
        
        3. Fine Attention Evaluation:
           $$A = \operatorname{Softmax}\left(\frac{Q_i K_{\text{cand}}^\top}{\sqrt{d_k}} + M\right)$$
           $$O_i = A V_{\text{cand}}$$

    Complexity:
        - Coarse Routing: $\mathcal{O}(B \cdot H \cdot M^2 \cdot d_k)$ where $M = N / B_s \ll N$.
        - Fine Attention: $\mathcal{O}(B \cdot H \cdot N \cdot (k \cdot B_s) \cdot d_k)$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        block_size: int = 32,
        sparsity: float = 0.5,
        pretrained_attn: Optional[nn.Module] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = getattr(config, "num_attention_heads", 12)
        hidden_size: int = getattr(config, "hidden_size", 768)
        self.head_dim: int = hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.block_size: int = getattr(config, "xattn_block_size", block_size)
        self.sparsity: float = getattr(config, "xattn_sparsity", sparsity)

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

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q = self.query(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K = self.key(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.value(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

        valid_mask = get_token_valid_mask(attention_mask, B, L, device)  # (B, L)

        pad_len: int = (self.block_size - (L % self.block_size)) % self.block_size
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
            valid_mask_pad = F.pad(valid_mask, (0, pad_len), value=False)
        else:
            valid_mask_pad = valid_mask

        padded_len: int = L + pad_len
        n_blocks: int = padded_len // self.block_size

        Q_blk = Q.view(B, H, n_blocks, self.block_size, d)
        K_blk = K.view(B, H, n_blocks, self.block_size, d)
        V_blk = V.view(B, H, n_blocks, self.block_size, d)
        mask_blk = valid_mask_pad.view(B, 1, n_blocks, self.block_size)

        scale = d ** -0.5

        # Coarse block-level summaries
        mask_blk_float = mask_blk.unsqueeze(-1).to(dtype)
        K_blk_masked = K_blk * mask_blk_float
        Q_mean = Q_blk.mean(dim=3)
        K_mean = K_blk_masked.sum(dim=3) / (mask_blk_float.sum(dim=3) + 1e-5)

        block_scores = torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale  # (B, H, n_blocks, n_blocks)
        num_keep = max(1, min(n_blocks, int(n_blocks * (1.0 - self.sparsity))))
        _, topk_idx = torch.topk(block_scores, num_keep, dim=-1)  # (B, H, n_blocks, num_keep)

        out_blocks = []
        for i in range(n_blocks):
            qi = Q_blk[:, :, i]  # (B, H, block_size, d)
            idx_i = topk_idx[:, :, i]  # (B, H, num_keep)

            idx_expanded = idx_i.unsqueeze(-1).unsqueeze(-1).expand(B, H, num_keep, self.block_size, d)
            k_cand = torch.gather(K_blk, 2, idx_expanded).view(B, H, num_keep * self.block_size, d)
            v_cand = torch.gather(V_blk, 2, idx_expanded).view(B, H, num_keep * self.block_size, d)

            mask_expanded = idx_i.unsqueeze(-1).expand(B, H, num_keep, self.block_size)
            cand_mask = torch.gather(mask_blk.expand(B, H, n_blocks, self.block_size), 2, mask_expanded).view(B, H, 1, num_keep * self.block_size)

            scores_i = torch.matmul(qi, k_cand.transpose(-2, -1)) * scale
            scores_i = scores_i.masked_fill(~cand_mask, -10000.0)

            attn_i = F.softmax(scores_i, dim=-1)
            attn_i = self.dropout(attn_i)
            ctx_i = torch.matmul(attn_i, v_cand)
            out_blocks.append(ctx_i)

        out_stacked = torch.stack(out_blocks, dim=2)  # (B, H, n_blocks, block_size, d)
        out = out_stacked.view(B, H, padded_len, d)[:, :, :L, :]
        if head_mask is not None:
            out = out * head_mask

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, self.all_head_size)
        out = self.out_proj(out)
        out = out * valid_mask.unsqueeze(-1).to(dtype)
        return out, None


@ATTENTION_REGISTRY.register("xattention_inject")
def inject_xattention(
    layer: nn.Module,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Optional[nn.Module],
) -> XAttention:
    block_size = getattr(comp_config, "xattn_block_size", 32)
    sparsity = getattr(comp_config, "xattn_sparsity", 0.5)
    new_attn = XAttention(
        config=model_config,
        block_size=block_size,
        sparsity=sparsity,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
