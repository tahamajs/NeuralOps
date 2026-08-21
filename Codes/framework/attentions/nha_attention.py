"""
Native Hybrid Attention (NHA) – Combines full attention for heavy-hitter tokens
with linear recurrent kernel attention for context tokens.
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("nha")
class NHAAttention(BaseAttention):
    r"""Native Hybrid Attention (NHA) Architecture.

    Partitions sequence tokens into high-salience heavy hitters $\mathcal{S}_{\text{top}}$
    (processed via full softmax attention) and remaining context tokens (processed via $\mathcal{O}(N)$
    positive kernel linear attention $\phi(x) = \text{ELU}(x) + 1$):
    $$\text{Attention}_{\text{linear}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$
    $$\text{Output}_i = \begin{cases} \text{FullAttention}(Q_i, K, V), & i \in \mathcal{S}_{\text{top}} \\ \text{LinearAttention}(Q_i, K, V), & i \notin \mathcal{S}_{\text{top}} \end{cases}$$

    Computational Complexity:
        - Heavy-Hitters: $\mathcal{O}(B \cdot H \cdot (\rho N) \cdot N \cdot d_k)$ where $\rho = \text{full\_ratio} \ll 1$
        - Linear Context: $\mathcal{O}(B \cdot H \cdot N \cdot d_k^2)$
        - Overall Time: $\mathcal{O}(B \cdot H \cdot (\rho N^2 + N d_k) d_k)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - attention_mask: Optional $[B, N]$ or $[B, 1, 1, N]$.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, N])$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        full_ratio: float = 0.1,
    ) -> None:
        super().__init__(config, rank)
        self.full_ratio: float = full_ratio
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.hidden_size: int = config.hidden_size

        self.q_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass combining selective full and positive-kernel linear attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask `[Batch, SeqLen]`.
            output_attentions (bool): Whether to return full attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation `[Batch, SeqLen, HiddenDim]` and optional attention map.
        """
        B, L, D = hidden_states.shape
        H, Dh = self.num_heads, self.head_dim

        q: torch.Tensor = (
            self.q_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(hidden_states).view(B, L, H, Dh).transpose(1, 2)
        )

        with torch.no_grad():
            token_score: torch.Tensor = q.norm(dim=-1).mean(dim=1)  # (B, L)
            num_full: int = max(1, int(L * self.full_ratio))
            _, top_indices = torch.topk(token_score, k=num_full, dim=-1)

        full_mask: torch.Tensor = torch.zeros(
            B, L, dtype=torch.bool, device=q.device
        )
        for b in range(B):
            full_mask[b, top_indices[b]] = True
        full_mask = (
            full_mask.unsqueeze(1).unsqueeze(-1).expand(-1, H, -1, Dh)
        )  # (B, H, L, Dh)

        full_q: torch.Tensor = q * full_mask
        full_k: torch.Tensor = k * full_mask
        scores: torch.Tensor = torch.matmul(full_q, full_k.transpose(-2, -1)) / (
            Dh**0.5
        )
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                scores = scores + (attention_mask.unsqueeze(1).unsqueeze(2).float() - 1.0) * -10000.0
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask
        attn_full: torch.Tensor = F.softmax(scores, dim=-1)
        attn_full = self.dropout(attn_full)
        full_out: torch.Tensor = torch.matmul(attn_full, v)

        q_lin: torch.Tensor = F.elu(q) + 1.0
        k_lin: torch.Tensor = F.elu(k) + 1.0
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)
            elif attention_mask.dim() == 4:
                mask_expanded = (attention_mask[:, 0, 0, :] > -10000.0).unsqueeze(1).unsqueeze(-1)
            else:
                mask_expanded = attention_mask.unsqueeze(1).unsqueeze(-1)
            mask_expanded = (mask_expanded.float() - 1.0) * -10000.0
            k_lin = k_lin + mask_expanded

        kv_lin: torch.Tensor = torch.einsum(
            "b h n d, b h n e -> b h d e", k_lin, v
        )
        z_lin: torch.Tensor = k_lin.sum(dim=2) + 1e-5
        denom: torch.Tensor = torch.einsum(
            "b h n d, b h d -> b h n", q_lin, z_lin
        ).unsqueeze(-1)
        lin_out: torch.Tensor = (
            torch.einsum("b h n d, b h d e -> b h n e", q_lin, kv_lin) / denom
        )

        out: torch.Tensor = torch.where(full_mask, full_out, lin_out)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn_full
        return out, None
