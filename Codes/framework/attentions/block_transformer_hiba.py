r"""
Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).

Implements block-partitioned attention consisting of:
1. Intra-block full causal attention within contiguous sequence blocks.
2. Inter-block sparse top-$k$ cross-block attention to maintain long-range dependencies.
3. Periodic full dense rectification passes to prevent attention fragmentation.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("block_transformer_hiba")
class BlockTransformerHIBA(BaseAttention):
    r"""
    Block Transformer with Hierarchical Intra-Block and Inter-Block Attention (HIBA).

    Mathematical Formulation:
        Given sequence length $L$ partitioned into $M = \lceil L / B_{\text{size}} \rceil$ blocks of size $B_{\text{size}}$:
        1. Intra-Block Causal Attention:
           For each block $b \in \{1, \dots, M\}$:
           $$O_b^{\text{intra}} = \text{Softmax}\left(\frac{Q_b K_b^T}{\sqrt{d_h}} + M_{\text{causal}}\right) V_b$$

        2. Inter-Block Sparse Top-$k$ Attention:
           Computes queries against out-of-block context $K_{\neg b}, V_{\neg b}$:
           $$S_{ij}^{\text{inter}} = \frac{(Q_b)_i (K_{\neg b})_j^T}{\sqrt{d_h}}$$
           $$\mathcal{I}_{\text{top-}k} = \text{Top-}k(S^{\text{inter}}, k_{\text{inter}})$$
           $$O_b^{\text{inter}} = \text{Softmax}\left( \text{Mask}_{\text{top-}k}(S^{\text{inter}}) \right) V_{\neg b}$$

        3. Periodic Dense Rectification:
           Every $T_{\text{rect}}$ layers, full global attention is evaluated:
           $$O = \text{Softmax}\left(\frac{Q K^T}{\sqrt{d_h}} + M_{\text{attn}}\right) V$$

    Complexity:
        - Intra/Inter Attention Pass: $\mathcal{O}\left(B \cdot H \cdot L \cdot (B_{\text{size}} + k_{\text{inter}}) \cdot d_h\right)$ time and memory.
        - Rectification Pass: $\mathcal{O}(B \cdot H \cdot L^2 \cdot d_h)$.

    Tensor Dimensions:
        - Input `hidden_states` / `x`: $[B, L, D]$
        - Block Query / Key / Value: $[B, H, B_{\text{size}}, d_h]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        block_size: int = 512,
        top_k_inter: int = 32,
        rectification_interval: int = 4,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.dim // self.num_heads
        self.block_size: int = block_size
        self.top_k_inter: int = top_k_inter
        self.rectification_interval: int = rectification_interval

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.out_proj = nn.Linear(self.dim, self.dim)

        self.layer_idx: int = 0

    def _intra_block_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        causal: bool = True,
    ) -> torch.Tensor:
        r"""
        Computes dense scaled dot-product attention within an isolated block.

        Args:
            q (torch.Tensor): Block query tensor $[B, H, L_b, d_h]$.
            k (torch.Tensor): Block key tensor $[B, H, L_b, d_h]$.
            v (torch.Tensor): Block value tensor $[B, H, L_b, d_h]$.
            causal (bool): Whether to enforce lower-triangular causal masking.

        Returns:
            torch.Tensor: Attended block representations $[B, H, L_b, d_h]$.
        """
        scale = self.head_dim**-0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if causal:
            L = q.size(2)
            mask = torch.tril(torch.ones(L, L, device=q.device)).bool()
            attn = attn.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        return torch.matmul(F.softmax(attn, dim=-1), v)

    def _inter_block_sparse(
        self,
        q_b: torch.Tensor,
        k_all: torch.Tensor,
        v_all: torch.Tensor,
        block_idx: int,
        num_blocks: int,
    ) -> torch.Tensor:
        r"""
        Computes top-$k$ sparse attention across non-local sequence blocks.

        Args:
            q_b (torch.Tensor): Current block queries $[B, H, L_b, d_h]$.
            k_all (torch.Tensor): Global key representations $[B, H, L, d_h]$.
            v_all (torch.Tensor): Global value representations $[B, H, L, d_h]$.
            block_idx (int): Current block index $b$.
            num_blocks (int): Total number of blocks $M$.

        Returns:
            torch.Tensor: Inter-block attended context $[B, H, L_b, d_h]$.
        """
        B, H, _, D = q_b.shape
        total_len = k_all.size(2)
        start_b = block_idx * self.block_size
        end_b = min((block_idx + 1) * self.block_size, total_len)

        k_select = []
        v_select = []
        if block_idx > 0:
            k_select.append(k_all[:, :, :start_b, :])
            v_select.append(v_all[:, :, :start_b, :])
        if (block_idx + 1) * self.block_size < total_len:
            k_select.append(k_all[:, :, end_b:, :])
            v_select.append(v_all[:, :, end_b:, :])

        if not k_select:
            return torch.zeros_like(q_b)
        k_neighbors = torch.cat(k_select, dim=2)
        v_neighbors = torch.cat(v_select, dim=2)

        scale = self.head_dim**-0.5
        scores = torch.matmul(q_b, k_neighbors.transpose(-2, -1)) * scale
        top_k = min(self.top_k_inter, k_neighbors.size(2))
        _, topk_idx = torch.topk(scores, top_k, dim=-1)
        mask = torch.zeros_like(scores).scatter_(-1, topk_idx, 1.0)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        return torch.matmul(F.softmax(scores, dim=-1), v_neighbors)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward execution of HIBA Block Transformer layer.

        Args:
            hidden_states (torch.Tensor): Input representation of shape $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor of shape $[B, L]$ or $[B, 1, 1, L]$.
            output_attentions (bool): Flag indicating whether to return attention matrices.
            **kwargs (Any): Additional configuration parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Transformed representations of shape $[B, L, D]$.
                - None (or attention weights).
        """
        x = hidden_states
        B, L, D = x.shape
        H = self.num_heads
        head_dim = self.head_dim

        q = self.W_q(x).view(B, L, H, head_dim).permute(0, 2, 1, 3)
        k = self.W_k(x).view(B, L, H, head_dim).permute(0, 2, 1, 3)
        v = self.W_v(x).view(B, L, H, head_dim).permute(0, 2, 1, 3)

        num_blocks = (L + self.block_size - 1) // self.block_size
        use_rectify = self.layer_idx % self.rectification_interval == 0
        self.layer_idx += 1

        if use_rectify:
            scale = head_dim**-0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if attention_mask is not None:
                if attention_mask.dim() == 2:
                    mask_4d = attention_mask.unsqueeze(1).unsqueeze(2).float()
                    scores = scores + (1.0 - mask_4d) * -10000.0
                elif attention_mask.dim() == 4:
                    scores = scores + attention_mask.float()
            attn_probs = F.softmax(scores, dim=-1)
            out = torch.matmul(attn_probs, v)
        else:
            out_list = []
            for b_idx in range(num_blocks):
                start = b_idx * self.block_size
                end = min(start + self.block_size, L)
                q_b = q[:, :, start:end, :]
                k_b = k[:, :, start:end, :]
                v_b = v[:, :, start:end, :]

                intra_out = self._intra_block_attention(q_b, k_b, v_b, causal=True)
                inter_out = self._inter_block_sparse(q_b, k, v, b_idx, num_blocks)
                out_b = intra_out + inter_out
                out_list.append(out_b)
            out = torch.cat(out_list, dim=2)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        return self.out_proj(out), None
