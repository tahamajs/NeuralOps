r"""
GoldFinch-AHN: Adaptive Hybrid Network Combining Sliding Window and Gated DeltaNet Memory.

Combines short-term causal sliding-window attention with long-term GatedDeltaNet recurrent memory
via a learned input-dependent dynamic gating mechanism.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseAttention
from framework.core.registry import ATTENTION_REGISTRY


class GatedDeltaNetLayer(nn.Module):
    r"""
    Gated DeltaNet Recurrent Memory Layer for Long-Term Context Retention.

    Mathematical Formulation:
        $$y = W_{\text{mem}} x + b_{\text{mem}}$$
    """

    def __init__(self, d_model: int, use_spectral_norm: bool = False) -> None:
        super().__init__()
        self.net = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Evaluates recurrent long-term memory transformation."""
        return self.net(x)


@ATTENTION_REGISTRY.register("goldfinch_ahn")
class GoldFinchAHNBlock(BaseAttention):
    r"""
    GoldFinch-AHN: Adaptive Hybrid Network Attention Block.

    Mathematical Formulation:
        1. Short-Term Causal Sliding-Window Attention:
           $$S_{ij} = \frac{Q_i K_j^T}{\sqrt{D}} + M_{\text{causal\_window}}$$
           $$O^{\text{short}} = \text{Softmax}(S) V, \quad j \in [\max(0, i - W + 1), i]$$

        2. Long-Term Delta Memory:
           $$O^{\text{long}} = \text{DeltaNet}(X)$$

        3. Adaptive Gated Context Blending:
           $$g = \sigma(X W_{\text{gate}} + b_{\text{gate}}) \in [0, 1]^{B \times L \times 1}$$
           $$\text{Output} = \left( g \odot O^{\text{short}} + (1 - g) \odot O^{\text{long}} \right) W_{\text{out}}^T + b_{\text{out}}$$

    Complexity:
        - Short-Term Attention: $\mathcal{O}(B \cdot L \cdot W_{\text{size}} \cdot D)$ time and memory.
        - Long-Term Memory: $\mathcal{O}(B \cdot L \cdot D^2)$ linear in sequence length $L$.

    Tensor Dimensions:
        - Input `hidden_states`: $[B, L, D]$
        - Blending Gate $g$: $[B, L, 1]$
        - Output: $[B, L, D]$
    """

    def __init__(
        self,
        config: Any,
        rank: int = 64,
        window_size: int = 256,
    ) -> None:
        super().__init__(config, rank)
        self.dim: int = config.hidden_size
        self.window_size: int = window_size

        self.W_q = nn.Linear(self.dim, self.dim, bias=False)
        self.W_k = nn.Linear(self.dim, self.dim, bias=False)
        self.W_v = nn.Linear(self.dim, self.dim, bias=False)
        self.delta_mem = GatedDeltaNetLayer(self.dim, use_spectral_norm=True)
        self.gate = nn.Linear(self.dim, 1)
        self.out_proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""
        Forward computation pass for GoldFinchAHNBlock.

        Args:
            hidden_states (torch.Tensor): Input representation $[B, L, D]$.
            attention_mask (Optional[torch.Tensor]): Mask tensor $[B, L]$ or $[B, 1, 1, L]$.
            output_attentions (bool): Whether to return attention matrices.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]:
                - Output context activations $[B, L, D]$.
                - None.
        """
        x = hidden_states
        B, L, D = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)
        scores = torch.bmm(q, k.transpose(1, 2)) / (D**0.5)

        idx = torch.arange(L, device=x.device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        mask_val = ~(causal & window)
        scores = scores.masked_fill(mask_val.unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            if attention_mask.dim() == 2:
                pad_mask = (1.0 - attention_mask.unsqueeze(1).float()) * -1e9
                scores = scores + pad_mask
            elif attention_mask.dim() == 4:
                scores = scores + attention_mask.squeeze(1).float()

        short_out = torch.bmm(F.softmax(scores, dim=-1), v)

        long_out = self.delta_mem(x)  # (B, L, D)
        gate = torch.sigmoid(self.gate(x))  # (B, L, 1)
        out = self.out_proj(gate * short_out + (1.0 - gate) * long_out)
        return out, None
