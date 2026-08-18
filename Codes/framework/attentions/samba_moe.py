r"""
SambaMoE: Hybrid Sliding-Window Attention, Mamba SSM, and Sparse Mixture-of-Experts.

Integrates sliding-window multi-head attention, Spectral Mamba state-space recurrence,
and top-$k$ sparse Mixture-of-Experts (MoE) feedforward transformations into a unified sub-quadratic block.

Mathematical Mechanism:
    Given input $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Sliding-Window Attention & Mamba Block:**
       $$X^{(1)} = \operatorname{SAMBA-M3}(X) \in \mathbb{R}^{B \times L \times D}$$
    2. **Top-$k$ Token-Choice MoE Routing:**
       For each token representation $x_i \in \mathbb{R}^D$:
       $$s_i = x_i W_{\text{router}} \in \mathbb{R}^{E}$$
       $$\mathcal{E}_i = \operatorname{arg\,top-}k(s_i), \quad w_{i, e} = \frac{\exp(s_{i, e})}{\sum_{e' \in \mathcal{E}_i} \exp(s_{i, e'})}$$
       $$y_i = \sum_{e \in \mathcal{E}_i} w_{i, e} \cdot \operatorname{Expert}_e(x_i)$$
    3. **Residual Normalization:**
       $$X^{(2)} = \operatorname{LayerNorm}(X^{(1)} + y)$$

Computational Complexity:
    - SAMBA-M3 stage: $\mathcal{O}(B \cdot L \cdot D \cdot (W + d_{\text{state}}))$.
    - Sparse MoE stage: $\mathcal{O}(B \cdot L \cdot k \cdot D \cdot d_{\text{ffn}})$.
    - Total block complexity: $\mathcal{O}(L \cdot D \cdot (W + d_{\text{state}} + k \cdot d_{\text{ffn}}))$, strictly linear $\mathcal{O}(L)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.registry import ATTENTION_REGISTRY

from .samba_m3 import SAMBAM3Block


@ATTENTION_REGISTRY.register("reap_moe")
class REAPMoE(nn.Module):
    r"""Sparse Token-Choice Mixture-of-Experts layer with top-$k$ gating.

    Args:
        d_model: Input/output feature dimension $D$.
        n_experts: Number of parallel expert networks ($E$).
        top_k: Number of active experts activated per token ($k$).
    """

    def __init__(
        self,
        d_model: Any = 768,
        n_experts: int = 8,
        top_k: int = 2,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)
        elif hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            top_k = getattr(config, "top_k", top_k)

        self.n_experts: int = n_experts
        self.top_k: int = top_k
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.GELU(),
                    nn.Linear(d_model * 4, d_model),
                )
                for _ in range(n_experts)
            ]
        )
        self.router = nn.Linear(d_model, n_experts)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Route tokens to top-$k$ experts and accumulate weighted outputs.

        Args:
            x: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            **kwargs: Additional keyword arguments.

        Returns:
            Tuple of (expert aggregated output tensor `[Batch, SeqLen, HiddenDim]`, None).
        """
        B, L, D = x.shape
        scores = self.router(x)
        weights, indices = torch.topk(scores, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        out = torch.zeros_like(x)
        for e in range(self.n_experts):
            mask = (indices == e).any(dim=-1)
            if mask.any():
                # FIX: [Cast mask indicator to weights.dtype to ensure dtype homogeneity in weighting]
                selected_weights = (weights * (indices == e).to(dtype=weights.dtype)).sum(
                    dim=-1, keepdim=True
                )
                out += selected_weights * self.experts[e](x * mask.unsqueeze(-1))
        return out, None


@ATTENTION_REGISTRY.register("samba_moe")
class SambaMoEBlock(nn.Module):
    r"""Composite SambaMoE block integrating SAMBA-M3 and REAPMoE.

    Args:
        d_model: Hidden feature dimension $D$.
        n_heads: Number of attention heads $H$.
        n_experts: Number of feedforward experts in the MoE pool ($E$).
        window: Sliding-window causal receptive field ($W$).
    """

    def __init__(
        self,
        d_model: Any = 768,
        n_heads: int = 12,
        n_experts: int = 8,
        window: int = 128,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            window = getattr(config, "window_size", window)
        elif hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            n_experts = getattr(config, "num_experts", getattr(config, "n_experts", n_experts))
            window = getattr(config, "window_size", window)

        self.swa_mamba = SAMBAM3Block(d_model, n_heads, window=window)
        self.moe = REAPMoE(d_model, n_experts)
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Execute SambaMoE forward pass.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Layer output tensor `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention matrix or None.
        """
        x, attn_w = self.swa_mamba(hidden_states, attention_mask=attention_mask, **kwargs)
        moe_out = self.moe(x)
        if isinstance(moe_out, tuple):
            moe_out = moe_out[0]
        x = self.norm(x + moe_out)
        return x, attn_w
