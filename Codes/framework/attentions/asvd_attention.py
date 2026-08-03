"""
ASVD Attention: Activation‑Aware Singular Value Decomposition for Attention Compression
"""

from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("asvd")
class ASVDAttention(BaseAttention):
    r"""Activation-Aware Singular Value Decomposition (ASVD) Attention.

    Compresses attention projection weight matrices $W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$
    by scaling weights by the empirical input channel activation magnitude $S_X = \text{diag}(\mathbf{s}_X)$,
    performing low-rank truncated SVD on the scaled weight $W S_X = U \Sigma V^\top$, and absorbing scales back:
    $$W = (U \Sigma) (V^\top S_X^{-1})$$
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d_k}} + M\right) V$$

    Computational Complexity:
        - Parameter Footprint: $\mathcal{O}(4 \cdot r \cdot (d + d_{\text{all}}))$
        - Time: $\mathcal{O}(B \cdot H \cdot N^2 \cdot d_k + B \cdot N \cdot d \cdot d_{\text{all}})$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - attention_mask: Optional $[B, N]$ or $[B, 1, 1, N]$.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        activation_scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> None:
        super().__init__(config, rank)
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim
        self.rank: int = rank

        self.query_u: nn.Parameter = nn.Parameter(
            torch.randn(self.all_head_size, self.rank) * 0.02
        )
        self.query_s: nn.Parameter = nn.Parameter(torch.ones(self.rank))
        self.query_vh: nn.Parameter = nn.Parameter(
            torch.randn(self.rank, config.hidden_size) * 0.02
        )

        self.key_u: nn.Parameter = nn.Parameter(
            torch.randn(self.all_head_size, self.rank) * 0.02
        )
        self.key_s: nn.Parameter = nn.Parameter(torch.ones(self.rank))
        self.key_vh: nn.Parameter = nn.Parameter(
            torch.randn(self.rank, config.hidden_size) * 0.02
        )

        self.value_u: nn.Parameter = nn.Parameter(
            torch.randn(self.all_head_size, self.rank) * 0.02
        )
        self.value_s: nn.Parameter = nn.Parameter(torch.ones(self.rank))
        self.value_vh: nn.Parameter = nn.Parameter(
            torch.randn(self.rank, config.hidden_size) * 0.02
        )

        self.out_proj_u: nn.Parameter = nn.Parameter(
            torch.randn(config.hidden_size, self.rank) * 0.02
        )
        self.out_proj_s: nn.Parameter = nn.Parameter(torch.ones(self.rank))
        self.out_proj_vh: nn.Parameter = nn.Parameter(
            torch.randn(self.rank, self.all_head_size) * 0.02
        )

        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        if activation_scales is not None:
            self.apply_activation_scales(activation_scales)

    def apply_activation_scales(
        self, scales: Optional[Union[torch.Tensor, np.ndarray]]
    ) -> None:
        """Apply activation scales to the right singular vectors for ASVD calibration."""
        if scales is not None:
            if not isinstance(scales, torch.Tensor):
                scales_t = torch.tensor(
                    scales, device=self.query_vh.device, dtype=self.query_vh.dtype
                )
            else:
                scales_t = scales
            if scales_t.dim() == 1:
                scales_t = scales_t.unsqueeze(0)
            self.query_vh.data.mul_(scales_t)
            self.key_vh.data.mul_(scales_t)
            self.value_vh.data.mul_(scales_t)
            self.out_proj_vh.data.mul_(scales_t)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for activation-aware decomposed attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Attention mask tensor.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation and optional attention weights.
        """
        batch_size, seq_len, _ = hidden_states.shape

        W_q: torch.Tensor = (self.query_u * self.query_s) @ self.query_vh
        W_k: torch.Tensor = (self.key_u * self.key_s) @ self.key_vh
        W_v: torch.Tensor = (self.value_u * self.value_s) @ self.value_vh
        W_o: torch.Tensor = (self.out_proj_u * self.out_proj_s) @ self.out_proj_vh

        Q: torch.Tensor = (
            F.linear(hidden_states, W_q, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        K: torch.Tensor = (
            F.linear(hidden_states, W_k, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        V: torch.Tensor = (
            F.linear(hidden_states, W_v, None)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )

        scale: float = 1.0 / (self.head_dim**0.5)
        scores: torch.Tensor = torch.matmul(Q, K.transpose(-2, -1)) * scale

        if attention_mask is not None:
            scores = scores + attention_mask.unsqueeze(1)

        probs: torch.Tensor = F.softmax(scores, dim=-1)
        probs = self.dropout(probs)

        context: torch.Tensor = torch.matmul(probs, V)
        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )

        out: torch.Tensor = F.linear(context, W_o)
        if kwargs.get("output_attentions"):
            return (out, probs)
        return (out, None)

    @staticmethod
    def from_module(
        module: Any,
        rank: int,
        activation_scales: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ) -> "ASVDAttention":
        """Create ASVDAttention by SVD of the original module's weights."""
        attn = ASVDAttention(module, rank, activation_scales)
        w_qkv = module.in_proj_weight.data.cpu().numpy()
        w_o = module.out_proj.weight.data.cpu().numpy()
        all_head = module.embed_dim * module.num_heads
        W_q = w_qkv[:all_head, :]
        W_k = w_qkv[all_head : 2 * all_head, :]
        W_v = w_qkv[2 * all_head :, :]

        def decomposed_weights(
            W: np.ndarray, rank: int
        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            U, S, Vh = np.linalg.svd(W, full_matrices=False)
            U_r = U[:, :rank]
            S_r = S[:rank]
            Vh_r = Vh[:rank, :]
            return (
                torch.tensor(U_r, dtype=torch.float32),
                torch.tensor(S_r, dtype=torch.float32),
                torch.tensor(Vh_r, dtype=torch.float32),
            )

        qu, qs, qvh = decomposed_weights(W_q, rank)
        ku, ks, kvh = decomposed_weights(W_k, rank)
        vu, vs, vvh = decomposed_weights(W_v, rank)
        ou, os, ovh = decomposed_weights(w_o, rank)

        attn.query_u.data = qu
        attn.query_s.data = qs
        attn.query_vh.data = qvh
        attn.key_u.data = ku
        attn.key_s.data = ks
        attn.key_vh.data = kvh
        attn.value_u.data = vu
        attn.value_s.data = vs
        attn.value_vh.data = vvh
        attn.out_proj_u.data = ou
        attn.out_proj_s.data = os
        attn.out_proj_vh.data = ovh
        return attn
