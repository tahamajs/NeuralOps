"""
Rank‑Augmented Linear Attention (RALA) – ICLR 2024.
Augments linear attention kernel maps with low-rank static head interaction matrices.
"""

from typing import Any, Optional, Tuple, Union

import torch
from framework.core.gqa_utils import get_qkv_projection_dims, repeat_kv
import torch.nn as nn
import torch.nn.functional as F

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("rala")
class RALALinearAttention(nn.Module):
    r"""Rank-Augmented Linear Attention (RALA) Mechanism.

    Augments standard kernel-based linear attention ($\mathcal{O}(N)$) with a low-rank parametric
    correction matrix $U_h V_h^\top \in \mathbb{R}^{d_k \times d_k}$ (rank $r \ll d_k$) to mitigate
    the capacity bottlenecks of standard low-rank kernel approximations:
    $$\text{Attention}_{\text{RALA}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})} + \alpha \frac{\phi(Q) (U V^\top)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot N \cdot d_k^2 + B \cdot H \cdot N \cdot d_k \cdot r)$
        - Memory: $\mathcal{O}(B \cdot H \cdot (d_k^2 + N \cdot d_k))$
        - Strictly linear in sequence length $N$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - attention_mask: Optional $[B, N]$ or $[B, 1, 1, N]$.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(self, config: Any,
        **kwargs) -> None:
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.hidden_size // self.num_heads
        self.scaling: float = self.head_dim**-0.5

        self.q_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(self.hidden_size, self.hidden_size)

        self.rank_r: int = getattr(config, "rala_rank", 8)
        self.U: nn.Parameter = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )
        self.V: nn.Parameter = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )

        self.alpha: nn.Parameter = nn.Parameter(torch.tensor(0.1))
        self.activation: nn.GELU = nn.GELU()

    @staticmethod
    def _apply_linear_mask(
        k: torch.Tensor, attention_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """ apply linear mask."""
        if attention_mask is None:
            return k
        
        # Ensure mask is [B, L]
        if attention_mask.dim() == 4:
            mask = attention_mask[:, 0, 0, :]
        elif attention_mask.dim() == 3:
            mask = attention_mask[:, 0, :]
        else:
            mask = attention_mask
            
        # Reshape to [B, 1, L, 1] so it can broadcast against k [B, H, L, d_k]
        mask = mask.unsqueeze(1).unsqueeze(-1)
        
        mask_val: torch.Tensor = (
            (mask > -10000.0).to(torch.float32)
        )
        return k * mask_val.to(k.dtype)

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
        """Forward pass for rank-augmented linear attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor `[Batch, SeqLen]`.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Contextualized representation `[Batch, SeqLen, HiddenDim]` and `None`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        q: torch.Tensor = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k: torch.Tensor = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v: torch.Tensor = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q_prime: torch.Tensor = F.elu(q) + 1.0
        k_prime: torch.Tensor = self._apply_linear_mask(
            F.elu(k) + 1.0, attention_mask
        )

        kv: torch.Tensor = torch.einsum("b h n d, b h n e -> b h d e", k_prime, v)
        z: torch.Tensor = k_prime.sum(dim=2)  # (B, H, D)

        denom: torch.Tensor = (
            torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1)
            + 1e-5
        )
        lin_out: torch.Tensor = (
            torch.einsum("b h n d, b h d e -> b h n e", q_prime, kv) / denom
        )

        rank_mat: torch.Tensor = torch.einsum(
            "h d r, h e r -> h d e", self.U, self.V
        )  # (H, D, E)
        aug_out: torch.Tensor = (
            torch.einsum("b h n d, h d e -> b h n e", q_prime, rank_mat) / denom
        )

        output: torch.Tensor = lin_out + self.alpha * aug_out
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        output = self.out_proj(output)

        return (output, None)
