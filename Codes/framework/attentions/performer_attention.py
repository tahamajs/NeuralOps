"""
Performer: Fast Attention Via Positive Orthogonal Random features (FAVOR+)
Paper: https://arxiv.org/abs/2009.14794
"""

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("performer")
class PerformerAttention(BaseAttention):
    r"""Performer Linear Attention via FAVOR+ Kernel Decomposition.

    Approximates softmax self-attention using Positive Orthogonal Random Features (FAVOR+):
    $$\phi(x) = \frac{h(x)}{\sqrt{M}} \exp\left(W x - \frac{\|x\|^2}{2}\right)$$
    where $W \in \mathbb{R}^{M \times d_k}$ are Gaussian orthogonal random features obtained via QR decomposition.
    $$\text{Attention}_{\text{Performer}}(Q, K, V) = \frac{\phi(Q) (\phi(K)^\top V)}{\phi(Q) (\phi(K)^\top \mathbf{1})}$$

    Computational Complexity:
        - Time: $\mathcal{O}(B \cdot H \cdot (N \cdot M \cdot d_k + N \cdot M))$
        - Memory: $\mathcal{O}(B \cdot H \cdot (N \cdot M + M \cdot d_k))$ vs $\mathcal{O}(B \cdot H \cdot N^2)$

    Tensor Dimensions:
        - hidden_states: $[B, N, D]$ where $B$ is batch size, $N$ is sequence length, and $D$ is hidden dimension.
        - Output: Tuple of $([B, N, D], \text{None})$.
    """

    def __init__(
        self,
        config: Any,
        rank: int = 256,
        redraw_projection: bool = False,
        pretrained_attn: Optional[nn.Module] = None,
    ) -> None:
        super().__init__(config, rank)
        self.rank: int = rank
        self.redraw_projection: bool = redraw_projection
        self.num_heads: int = config.num_attention_heads
        self.head_dim: int = config.hidden_size // self.num_heads
        self.all_head_size: int = self.num_heads * self.head_dim

        self.query: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.key: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.value: nn.Linear = nn.Linear(config.hidden_size, self.all_head_size)
        self.out_proj: nn.Linear = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout: nn.Dropout = nn.Dropout(config.dropout)

        self.register_buffer("projection", None)

        if pretrained_attn is not None:
            self.copy_weights(pretrained_attn)

    def copy_weights(self, src: Any) -> None:
        """Copy pretrained weights and biases."""
        with torch.no_grad():
            self.query.weight.copy_(src.q_lin.weight)
            if src.q_lin.bias is not None:
                self.query.bias.copy_(src.q_lin.bias)
            self.key.weight.copy_(src.k_lin.weight)
            if src.k_lin.bias is not None:
                self.key.bias.copy_(src.k_lin.bias)
            self.value.weight.copy_(src.v_lin.weight)
            if src.v_lin.bias is not None:
                self.value.bias.copy_(src.v_lin.bias)
            self.out_proj.weight.copy_(src.out_lin.weight)
            if src.out_lin.bias is not None:
                self.out_proj.bias.copy_(src.out_lin.bias)

    def _reshape(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        return x.view(batch_size, -1, self.num_heads, self.head_dim).permute(
            0, 2, 1, 3
        )

    def _get_projection_matrix(self, device: torch.device) -> torch.Tensor:
        """Create per‑head orthogonal random features (FAVOR+)."""
        if self.projection is None or self.redraw_projection:
            proj = []
            for _ in range(self.num_heads):
                ortho = torch.randn(self.head_dim, self.rank, device=device)
                q, _ = torch.linalg.qr(ortho)
                proj.append(q)
            self.projection = torch.stack(proj, dim=0)  # (H, D, R)
        return self.projection

    def _feature_map(self, x: torch.Tensor, proj: torch.Tensor) -> torch.Tensor:
        r"""Evaluate positive orthogonal random feature map: $\phi(x) = \exp(x R - \|x\|^2/2)/\sqrt{r}$."""
        x_proj: torch.Tensor = torch.einsum(
            "bhnd,hdr->bhnr", x, proj
        )  # (B, H, N, R)
        x_norm: torch.Tensor = (x * x).sum(dim=-1, keepdim=True) / 2.0
        phi: torch.Tensor = torch.exp(x_proj - x_norm) / math.sqrt(self.rank)
        return phi

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for FAVOR+ linear attention.

        Args:
            hidden_states (torch.Tensor): Input tensor `[Batch, SeqLen, HiddenDim]`.
            attention_mask (Optional[torch.Tensor]): Mask tensor.
            head_mask (Optional[torch.Tensor]): Head mask tensor.
            output_attentions (bool): Whether to output attention weights.
            **kwargs (Any): Additional parameters.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: Output features and `None`.
        """
        batch_size, seq_len, _ = hidden_states.shape
        Q: torch.Tensor = self._reshape(self.query(hidden_states), batch_size)
        K: torch.Tensor = self._reshape(self.key(hidden_states), batch_size)
        V: torch.Tensor = self._reshape(self.value(hidden_states), batch_size)

        proj: torch.Tensor = self._get_projection_matrix(Q.device)

        Q_feat: torch.Tensor = self._feature_map(Q, proj)
        K_feat: torch.Tensor = self._feature_map(K, proj)

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                valid = attention_mask.bool().to(Q.device)
            else:
                valid = attention_mask.squeeze(1).squeeze(1).bool()
            valid_expanded = valid.unsqueeze(1).unsqueeze(-1)
            K_feat = K_feat * valid_expanded.to(K_feat.dtype)
            V = V * valid_expanded.to(V.dtype)

        KV: torch.Tensor = torch.einsum(
            "bhnr,bhnd->bhrd", K_feat, V
        )  # (B, H, R, D)
        K_sum: torch.Tensor = K_feat.sum(dim=2, keepdim=True)  # (B, H, 1, R)
        num: torch.Tensor = torch.einsum("bhnr,bhrd->bhnd", Q_feat, KV)
        den: torch.Tensor = (
            torch.einsum("bhnr,bhr->bhn", Q_feat, K_sum.squeeze(2)).unsqueeze(
                -1
            )
            + 1e-8
        )
        context: torch.Tensor = num / den

        if attention_mask is not None:
            valid_out = valid.unsqueeze(1).unsqueeze(-1).to(context.dtype)
            context = context * valid_out

        context = (
            context.permute(0, 2, 1, 3)
            .contiguous()
            .view(batch_size, seq_len, self.all_head_size)
        )
        out: torch.Tensor = self.out_proj(context)

        return out, None


def inject_performer(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> PerformerAttention:
    """Injection helper for Performer attention."""
    rank: int = getattr(comp_config, "rank", 256)
    redraw: bool = getattr(comp_config, "redraw_projection", False)
    new_attn: PerformerAttention = PerformerAttention(
        config=model_config,
        rank=rank,
        redraw_projection=redraw,
        pretrained_attn=old_attn,
    )
    layer.attention = new_attn
    return new_attn
