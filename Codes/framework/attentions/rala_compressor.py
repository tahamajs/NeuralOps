r"""
Rank-Augmented Linear Attention (RALA).

Linear attention mechanism augmented with a low-rank parametric state memory matrix ($U V^T$)
to enhance representational capacity without sacrificing linear $\mathcal{O}(N)$ computational complexity.

Mathematical Mechanism:
    Let feature map $\phi(x) = \operatorname{ELU}(x) + 1.0 > 0$. For queries $Q \in \mathbb{R}^{B \times H \times N \times d}$, keys $K \in \mathbb{R}^{B \times H \times N \times d}$, and values $V \in \mathbb{R}^{B \times H \times N \times d}$:
    1. **Linear Associative Memory:**
       $$S_{\text{KV}} = \sum_{j=1}^N \phi(K_j) V_j^T \in \mathbb{R}^{B \times H \times d \times d}, \quad z = \sum_{j=1}^N \phi(K_j) \in \mathbb{R}^{B \times H \times d}$$
       $$O_{\text{lin}} = \frac{\phi(Q) S_{\text{KV}}}{\phi(Q) z^T + \epsilon} \in \mathbb{R}^{B \times H \times N \times d}$$
    2. **Rank-Augmented Parametric Projection:**
       With learnable low-rank factor matrices $U, V \in \mathbb{R}^{H \times d \times r}$ and mixing coefficient $\alpha$:
       $$M_{\text{rank}} = U V^T \in \mathbb{R}^{H \times d \times d}$$
       $$O_{\text{aug}} = \frac{\phi(Q) M_{\text{rank}}}{\phi(Q) z^T + \epsilon} \in \mathbb{R}^{B \times H \times N \times d}$$
    3. **Fused Output:**
       $$O = O_{\text{lin}} + \alpha \cdot O_{\text{aug}}$$

Computational Complexity:
    - Kernel evaluation & accumulation: $\mathcal{O}(B \cdot H \cdot N \cdot d^2)$.
    - Low-rank state projection: $\mathcal{O}(B \cdot H \cdot N \cdot d \cdot r)$.
    - Overall Complexity: $\mathcal{O}(N \cdot d \cdot (d + r))$, strictly linear $\mathcal{O}(N)$ in sequence length.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("rala_compressor")
class RALALinearAttention(nn.Module):
    r"""Rank-Augmented Linear Attention layer with positive ELU kernel features.

    Args:
        config: Model configuration specifying `hidden_size`, `num_attention_heads`, and optional `rala_rank`.
    """

    def __init__(self, config: Any) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = self.hidden_size // self.num_heads
        self.scaling: float = self.head_dim**-0.5

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.rank_r: int = int(getattr(config, "rala_rank", 8))
        self.U = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )
        self.V = nn.Parameter(
            torch.randn(self.num_heads, self.head_dim, self.rank_r) * 0.02
        )
        self.alpha = nn.Parameter(torch.tensor(0.1))
        self.activation = nn.GELU()

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
        r"""Execute RALA linear attention forward pass.

        Args:
            hidden_states: Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor `[Batch, SeqLen]` or broadcastable.
            output_attentions: Unused in linear attention; returns None for weights.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Projected tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: None.
        """
        batch_size, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        q_prime = F.elu(q) + 1.0
        k_prime = F.elu(k) + 1.0

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask.unsqueeze(1).unsqueeze(-1).to(k_prime.dtype)
            elif attention_mask.dim() == 3:
                mask = attention_mask.unsqueeze(-1).to(k_prime.dtype)
            elif attention_mask.dim() == 4:
                mask = (attention_mask > -10000.0).to(k_prime.dtype)
            else:
                mask = attention_mask.to(k_prime.dtype)
            k_prime = k_prime * mask

        kv = torch.einsum("b h n d, b h n e -> b h d e", k_prime, v)
        z = k_prime.sum(dim=2)
        denom = torch.einsum("b h n d, b h d -> b h n", q_prime, z).unsqueeze(-1) + 1e-5
        lin_out = torch.einsum("b h n d, b h d e -> b h n e", q_prime, kv) / denom

        rank_mat = torch.einsum("h d r, h e r -> h d e", self.U, self.V)
        aug_out = torch.einsum("b h n d, h d e -> b h n e", q_prime, rank_mat) / denom

        output = lin_out + self.alpha * aug_out
        output = (
            output.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )
        output = self.out_proj(output)
        return output, None
