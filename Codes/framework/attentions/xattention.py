"""XAttention: Block Sparse Attention with Coarse-to-Fine Routing.

Paper: https://arxiv.org/abs/2305.02102
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.attention_mask import get_token_valid_mask
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("xattention")
class XAttention(nn.Module):
    r"""XAttention: Block-Sparse Attention Mechanism with Unified Probability Normalization.

    Partitions sequence $N$ into $M = \lceil N / B_s \rceil$ non-overlapping blocks of size $B_s$.
    Evaluates coarse block-level interaction scores $S_{ij}^{\text{block}} = \frac{\bar{Q}_i \bar{K}_j^\top}{\sqrt{d_k}}$
    and gathers top-$k$ key/value blocks ($k = \max(1, \lfloor(1-s) M\rfloor)$) for each query block,
    evaluating a unified normalized softmax over all candidate tokens:
    $$K_{\text{cand}}^{(i)} = \operatorname{Concat}_{j \in \text{TopK}(S_{i,:}^{\text{block}})} K_j \in \mathbb{R}^{B \times H \times (k \cdot B_s) \times d}$$
    $$\text{Attention}_{\text{XAttn}}(Q_i, K, V) = \text{softmax}\left(\frac{Q_i (K_{\text{cand}}^{(i)})^\top}{\sqrt{d_k}} + M_{\text{cand}}^{(i)}\right) V_{\text{cand}}^{(i)}$$

    Computational Complexity:
        - Coarse Routing: $\mathcal{O}(B \cdot H \cdot M^2 \cdot d_k)$ where $M = N / B_s \ll N$.
        - Fine Attention: $\mathcal{O}(B \cdot H \cdot N \cdot (k \cdot B_s) \cdot d_k)$ vs $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        block_size: int = 64,
        sparsity: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.block_size: int = getattr(config, "xattn_block_size", block_size)
        self.sparsity: float = getattr(config, "xattn_sparsity", sparsity)

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        dropout_p = getattr(
            config,
            "dropout",
            getattr(
                config,
                "hidden_dropout_prob",
                getattr(config, "attention_probs_dropout_prob", 0.1),
            ),
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for block-sparse XAttention with unified normalization."""
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        device = hidden_states.device
        dtype = hidden_states.dtype

        Q: torch.Tensor = self.query(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        K: torch.Tensor = self.key(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        V: torch.Tensor = self.value(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)

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

        Q_blk: torch.Tensor = Q.view(B, H, n_blocks, self.block_size, d)
        K_blk: torch.Tensor = K.view(B, H, n_blocks, self.block_size, d)
        V_blk: torch.Tensor = V.view(B, H, n_blocks, self.block_size, d)
        mask_blk: torch.Tensor = valid_mask_pad.view(B, 1, n_blocks, self.block_size)

        scale: float = d**-0.5

        # Coarse block-level summaries
        # Mask out padding tokens inside block summaries
        mask_blk_float = mask_blk.unsqueeze(-1).to(dtype)
        K_blk_masked = K_blk * mask_blk_float
        Q_mean: torch.Tensor = Q_blk.mean(dim=3)  # (B, H, n_blocks, d)
        K_mean: torch.Tensor = K_blk_masked.sum(dim=3) / (mask_blk_float.sum(dim=3) + 1e-5)

        block_scores: torch.Tensor = (
            torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale
        )  # (B, H, n_blocks, n_blocks)

        num_keep: int = max(1, min(n_blocks, int(n_blocks * (1.0 - self.sparsity))))
        _, topk_idx = torch.topk(block_scores, num_keep, dim=-1)  # (B, H, n_blocks, num_keep)

        # Gathers candidate key/value blocks for each query block
        # topk_idx: (B, H, n_blocks, num_keep)
        out_blocks = []
        for i in range(n_blocks):
            qi = Q_blk[:, :, i]  # (B, H, block_size, d)
            idx_i = topk_idx[:, :, i]  # (B, H, num_keep)

            # Gather top-k key and value blocks
            # idx_expanded for K_blk: (B, H, num_keep, block_size, d)
            idx_expanded = idx_i.unsqueeze(-1).unsqueeze(-1).expand(B, H, num_keep, self.block_size, d)
            k_cand = torch.gather(K_blk, 2, idx_expanded).view(B, H, num_keep * self.block_size, d)
            v_cand = torch.gather(V_blk, 2, idx_expanded).view(B, H, num_keep * self.block_size, d)

            # Candidate mask
            mask_expanded = idx_i.unsqueeze(-1).expand(B, H, num_keep, self.block_size)
            cand_mask = torch.gather(mask_blk.expand(B, H, n_blocks, self.block_size), 2, mask_expanded).view(B, H, 1, num_keep * self.block_size)

            scores_i = torch.matmul(qi, k_cand.transpose(-2, -1)) * scale  # (B, H, block_size, num_keep * block_size)
            scores_i = scores_i.masked_fill(~cand_mask, -10000.0)

            attn_i = F.softmax(scores_i, dim=-1)
            attn_i = self.dropout(attn_i)
            ctx_i = torch.matmul(attn_i, v_cand)  # (B, H, block_size, d)
            out_blocks.append(ctx_i)

        out_stacked = torch.stack(out_blocks, dim=2)  # (B, H, n_blocks, block_size, d)
        out: torch.Tensor = out_stacked.view(B, H, padded_len, d)[:, :, :L, :]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        # Zero out padding tokens in output
        out = out * valid_mask.unsqueeze(-1).to(dtype)
        return out, None
