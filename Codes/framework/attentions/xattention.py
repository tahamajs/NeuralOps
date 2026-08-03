"""XAttention: Block Sparse with Antidiagonal Scoring.

Paper: https://arxiv.org/abs/2305.02102
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("xattention")
class XAttention(nn.Module):
    r"""XAttention: Block-Sparse Attention Mechanism with Coarse-Grained Routing.

    Partitions sequence $N$ into $M = \lceil N / B_s \rceil$ non-overlapping blocks of size $B_s$.
    Evaluates coarse block-level interaction scores $S_{ij}^{\text{block}} = \frac{1}{B_s^2} \sum_{u, v} Q_{i,u} K_{j,v}^\top / \sqrt{d_k}$
    and selectively computes fine-grained full attention only on top-$k$ blocks ($k = (1-s) M$ where $s$ is sparsity):
    $$\text{Attention}_{\text{XAttn}}(Q, K, V) = \sum_{j \in \text{TopK}(S_{i,:}^{\text{block}})} \text{softmax}\left(\frac{Q_i K_j^\top}{\sqrt{d_k}}\right) V_j$$

    Computational Complexity:
        - Coarse Routing: $\mathcal{O}(B \cdot H \cdot M^2 \cdot d_k)$ where $M = N / B_s \ll N$.
        - Fine Attention: $\mathcal{O}(B \cdot H \cdot (1-s) N \cdot B_s \cdot d_k)$ vs $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k)$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        block_size: int = 64,
        sparsity: float = 0.5,
    ) -> None:
        super().__init__()
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.block_size: int = getattr(config, "xattn_block_size", block_size)
        self.sparsity: float = getattr(config, "xattn_sparsity", sparsity)
        self.q_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for block-sparse XAttention.

        Args:
            hidden_states (torch.Tensor): Input hidden states `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Reconstructed features and `None`.
        """
        B, L, D = hidden_states.shape
        H, d = self.num_heads, self.head_dim
        Q: torch.Tensor = (
            self.q_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )
        K: torch.Tensor = (
            self.k_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )
        V: torch.Tensor = (
            self.v_proj(hidden_states).view(B, L, H, d).permute(0, 2, 1, 3)
        )

        pad_len: int = (self.block_size - (L % self.block_size)) % self.block_size
        if pad_len > 0:
            Q = F.pad(Q, (0, 0, 0, pad_len))
            K = F.pad(K, (0, 0, 0, pad_len))
            V = F.pad(V, (0, 0, 0, pad_len))
        padded_len: int = L + pad_len
        n_blocks: int = padded_len // self.block_size

        Q_blk: torch.Tensor = Q.view(B, H, n_blocks, self.block_size, d)
        K_blk: torch.Tensor = K.view(B, H, n_blocks, self.block_size, d)
        V_blk: torch.Tensor = V.view(B, H, n_blocks, self.block_size, d)

        scale: float = d**-0.5
        Q_mean: torch.Tensor = Q_blk.mean(dim=3)
        K_mean: torch.Tensor = K_blk.mean(dim=3)
        block_scores: torch.Tensor = (
            torch.matmul(Q_mean, K_mean.transpose(-2, -1)) * scale
        )

        num_keep: int = max(1, int(n_blocks * (1 - self.sparsity)))
        _, topk_idx = torch.topk(block_scores, num_keep, dim=-1)

        mask: torch.Tensor = torch.zeros_like(block_scores)
        mask.scatter_(-1, topk_idx, 1.0)

        out_blocks: torch.Tensor = torch.zeros(
            B, H, n_blocks, self.block_size, d, device=Q.device, dtype=Q.dtype
        )
        for i in range(n_blocks):
            qi = Q_blk[:, :, i]
            for j in range(n_blocks):
                if mask[:, :, i, j].sum() == 0:
                    continue
                kj = K_blk[:, :, j]
                vj = V_blk[:, :, j]
                scores = torch.matmul(qi, kj.transpose(-2, -1)) * scale
                attn = F.softmax(scores, dim=-1)
                attn = self.dropout(attn)
                ctx = torch.matmul(attn, vj)
                out_blocks[:, :, i] += ctx

        out: torch.Tensor = out_blocks.view(B, H, padded_len, d)[:, :, :L, :]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None
