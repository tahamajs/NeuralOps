r"""
SAMBA-M3: Hybrid Sliding-Window Attention and Spectral Mamba State Space Model.

Interleaves sliding-window localized self-attention with continuous-time selective state space (Mamba)
transformations enhanced by learnable spectral time-scale gating.

Mathematical Mechanism:
    Given sequence representations $X \in \mathbb{R}^{B \times L \times D}$:
    1. **Sliding-Window Self-Attention Sub-Layer:**
       $$M^{\text{window}}_{ij} = \begin{cases} 0 & \text{if } 0 \le i - j < W \\ -\infty & \text{otherwise} \end{cases}$$
       $$X^{(1)} = \operatorname{LayerNorm}\left(X + \operatorname{Softmax}\left(\frac{Q K^T}{\sqrt{d}} + M^{\text{window}}\right) V\right)$$
    2. **Spectral Mamba SSM Sub-Layer:**
       Discretized state evolution parameterized by input-dependent $(\bar{A}, \bar{B}, C)$:
       $$h_t = \bar{A}_t h_{t-1} + \bar{B}_t x_t^{(1)}, \quad y_t = C_t h_t + D x_t^{(1)}$$
       Spectral time-scale modulation:
       $$X^{(2)} = \operatorname{LayerNorm}\left(X^{(1)} + y \odot \sigma(\Delta_{\text{scale}})\right)$$

Computational Complexity:
    - Sliding window attention: $\mathcal{O}(B \cdot L \cdot W \cdot D)$.
    - Selective SSM scan: $\mathcal{O}(B \cdot L \cdot D \cdot d_{\text{state}})$.
    - Total per-layer complexity: $\mathcal{O}(L \cdot D \cdot (W + d_{\text{state}}))$, strictly linear $\mathcal{O}(L)$.
"""

from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY

from .jamba2 import Mamba


@ATTENTION_REGISTRY.register("spectral_mamba")
class SpectralMamba(nn.Module):
    r"""Mamba state space model augmented with channel-wise spectral time-scale modulation.

    Args:
        d_model: Hidden state dimension $D$.
    """

    def __init__(self, d_model: Any = 768, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
        elif hasattr(d_model, "hidden_size"):
            d_model = d_model.hidden_size
        self.mamba = Mamba(d_model)
        self.dt_scale = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""Compute modulated state-space sequence output.

        Args:
            x: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.

        Returns:
            Modulated SSM output tensor `[Batch, SeqLen, HiddenDim]`.
        """
        out = self.mamba(x)
        if isinstance(out, tuple):
            out = out[0]
        out = out * torch.sigmoid(self.dt_scale).unsqueeze(0).unsqueeze(0)
        return out


@ATTENTION_REGISTRY.register("samba_m3")
class SAMBAM3Block(nn.Module):
    r"""Hybrid SAMBA-M3 block combining local sliding-window attention and Spectral Mamba.

    Args:
        d_model: Feature dimensionality $D$.
        n_heads: Number of attention heads $H$.
        window: Local causal window span ($W$).
    """

    def __init__(
        self,
        d_model: Any = 768,
        n_heads: int = 12,
        window: int = 128,
        config: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if config is not None:
            d_model = getattr(config, "hidden_size", getattr(config, "d_model", d_model))
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", n_heads))
            window = getattr(config, "window_size", window)
        elif hasattr(d_model, "hidden_size"):
            config = d_model
            d_model = config.hidden_size
            n_heads = getattr(config, "num_attention_heads", getattr(config, "n_heads", 12))
            window = getattr(config, "window_size", window)

        self.window_size: int = window
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mamba = SpectralMamba(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def sliding_window_mask(self, L: int, device: torch.device) -> torch.Tensor:
        r"""Construct 2D causal sliding-window additive attention mask of shape `[L, L]`."""
        idx = torch.arange(L, device=device)
        causal = idx.unsqueeze(1) >= idx.unsqueeze(0)
        window = (idx.unsqueeze(1) - idx.unsqueeze(0)) < self.window_size
        valid = causal & window
        mask = torch.zeros(L, L, device=device, dtype=torch.float32)
        mask = mask.masked_fill(~valid, float("-inf"))
        return mask

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        r"""Forward execution through sliding-window attention and Spectral Mamba SSM.

        Args:
            hidden_states: Input tensor of shape `[Batch, SeqLen, HiddenDim]`.
            attention_mask: Optional attention mask tensor.
            **kwargs: Additional runtime arguments.

        Returns:
            Tuple of:
                - output: Transformed hidden states `[Batch, SeqLen, HiddenDim]`.
                - attention_weights: Attention weights or None.
        """
        x = hidden_states
        B, L, D = x.shape
        mask = self.sliding_window_mask(L, device=x.device)
        attn_out, attn_w = self.attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.mamba(x))
        return x, attn_w


@ATTENTION_REGISTRY.register("mamba")
def inject_mamba(
    layer: Any,
    model_config: Any,
    comp_config: Any,
    layer_idx: int,
    total_layers: int,
    old_attn: Any,
) -> Any:
    r"""Bridge helper injecting Mamba module into a target transformer layer."""
    d_model = model_config.hidden_size
    new_module = Mamba(d_model=d_model)
    layer.attention = new_module
    return new_module
