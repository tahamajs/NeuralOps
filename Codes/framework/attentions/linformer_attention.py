"""
Linformer: Linear‑complexity self‑attention via sequence‑length projection.
Paper: https://arxiv.org/abs/2006.04768
"""

from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("linformer")
class LinformerAttention(BaseAttention):
    r"""Linformer Linear-Complexity Attention via Sequence Dimension Projection.

    Projects the Key and Value sequence dimensions from $N$ to a low-dimensional fixed subspace
    $k \ll N$ using learnable linear projections $E, F \in \mathbb{R}^{k \times N}$:
    $$K_{\text{proj}} = E K \in \mathbb{R}^{B \times H \times k \times d_k}, \quad V_{\text{proj}} = F V \in \mathbb{R}^{B \times H \times k \times d_k}$$
    $$\text{Attention}_{\text{Linformer}}(Q, K, V) = \text{softmax}\left(\frac{Q K_{\text{proj}}^\top}{\sqrt{d_k}} + M\right) V_{\text{proj}}$$

    Computational Complexity:
        - Attention Matrix: $\mathcal{O}(B \cdot H \cdot N \cdot k)$
        - Context Product: $\mathcal{O}(B \cdot H \cdot N \cdot k \cdot d_k)$
        - Overall Time & Memory: Strictly linear $\mathcal{O}(N \cdot k \cdot d)$ vs standard $\mathcal{O}(N^2 d)$.

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{Optional}[B, H, N, k])$.
    """

    def __init__(
        self,
        config: Any,
        rank: Optional[int] = None,
        k_dim: int = 128,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config, rank)
        self.k_dim: int = k_dim
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads

        self.q_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.k_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.v_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_proj: nn.Linear = nn.Linear(config.hidden_size, config.hidden_size)
        dropout_p = getattr(config, "dropout", getattr(config, "hidden_dropout_prob", getattr(config, "attention_probs_dropout_prob", 0.1)))
        self.dropout: nn.Dropout = nn.Dropout(dropout_p)

        max_len: int = getattr(config, "max_position_embeddings", 512)
        self.E: nn.Parameter = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)
        self.F: nn.Parameter = nn.Parameter(torch.randn(k_dim, max_len) * 0.02)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy pretrained Q/K/V/O weights and biases from source attention module."""
        with torch.no_grad():
            self.q_proj.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.q_proj.bias.copy_(src.q_lin.bias)
            self.k_proj.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.k_proj.bias.copy_(src.k_lin.bias)
            self.v_proj.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.v_proj.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for low-rank Linformer attention.

        Args:
            hidden_states (torch.Tensor): Input activations of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Binary or additive mask tensor.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to return attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output representations `[Batch, SeqLen, HiddenDim]` and optional weights.
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

        E_eff: torch.Tensor = self.E[:, :L]
        F_eff: torch.Tensor = self.F[:, :L]
        k_proj: torch.Tensor = torch.einsum(
            "kl,bhld->bhkd", E_eff, k
        )  # (B, H, k_dim, Dh)
        v_proj: torch.Tensor = torch.einsum(
            "kl,bhld->bhkd", F_eff, v
        )  # (B, H, k_dim, Dh)

        scale: float = Dh**-0.5
        scores: torch.Tensor = (
            torch.matmul(q, k_proj.transpose(-2, -1)) * scale
        )  # (B, H, L, k_dim)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                mask = attention_mask[:, None, None, :].to(dtype=scores.dtype)
                scores = scores + (1.0 - mask) * -1e9
            else:
                scores = scores + attention_mask

        attn: torch.Tensor = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out: torch.Tensor = torch.matmul(attn, v_proj)
        out = out.transpose(1, 2).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if output_attentions:
            return out, attn
        return out, None


def inject_linformer(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> LinformerAttention:
    """Injection helper for Linformer attention."""
    k_dim: int = getattr(comp_config, "k_dim", 128)
    new_attn: LinformerAttention = LinformerAttention(
        config=model_config, k_dim=k_dim, pretrained_attn=old_attn
    )
    layer.attention = new_attn
    return new_attn
