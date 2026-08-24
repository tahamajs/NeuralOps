r"""
Universal Rotary Position Embedding (UnifiedRoPE).

Applies the same rotary positional encoding across both attention and SSM branches
in hybrid transformer architectures to guarantee positional consistency.

Mathematical Mechanism:
    For input $x \in \mathbb{R}^{B \times L \times D}$ with $D/2$ frequency bands:
    1. **Inverse Frequency Precomputation:**
       $$\theta_i = \text{base}^{-2i/D}, \quad i \in \{0, 1, \dots, D/2 - 1\}$$
    2. **Positional Embedding Cache:**
       $$\text{freqs}_{m, i} = m \cdot \theta_i \quad \forall m \in \{0, \dots, L_{\max}-1\}$$
       $$\text{cos\_cache} = \cos([\text{freqs}, \text{freqs}]), \quad \text{sin\_cache} = \sin([\text{freqs}, \text{freqs}]) \in \mathbb{R}^{L_{\max} \times D}$$
    3. **Rotation Application (Halved Dimensions):**
       $$x_{\text{rot}} = x \odot \cos + \text{rotate\_half}(x) \odot \sin$$
       where $\text{rotate\_half}(x) = [-x_{D/2:}, x_{:D/2}]$.

Computational Complexity:
    - Cache construction: $\mathcal{O}(L_{\max} \cdot D)$.
    - Forward rotation: $\mathcal{O}(B \cdot L \cdot D)$.
    - Space: $\mathcal{O}(L_{\max} \cdot D)$ for cos/sin buffers.
"""

from typing import Optional

import torch
import torch.nn as nn

from framework.core.registry import ATTENTION_REGISTRY


@ATTENTION_REGISTRY.register("compression_unified_rope")
class UnifiedRoPE(nn.Module):
    r"""Universal Rotary Position Embedding module.

    Applies consistent RoPE rotations suitable for both attention query-key
    pairs and state-space hidden representations.

    Args:
        dim: Feature dimension $D$ (must be even).
        max_seq_len: Maximum sequence length for precomputed cache.
        base: Frequency base for inverse frequency computation (default 10000).
    """

    def __init__(self, dim: Optional[int] = None, max_seq_len: int = 4096, base: int = 10000, config: Optional[Any] = None, **kwargs: Any) -> None:
        super().__init__()
        if dim is None:
            dim = getattr(config, "hidden_size", 768) if config is not None else 768
        self.dim: int = dim
        self.max_seq_len: int = max_seq_len
        self.base: int = base

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int) -> None:
        r"""Precompute cos & sin frequency tables for positions up to ``seq_len``."""
        t = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        r"""Apply RoPE rotational encoding to input tensor.

        Args:
            x: Input tensor of shape `[Batch, SeqLen, Dim]`.
            **kwargs: Additional keyword arguments (ignored).

        Returns:
            Rotationally encoded tensor of same shape `[Batch, SeqLen, Dim]`.
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)  # (1, L, 1, D)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        x_rot = x * cos.squeeze(2) + self._rotate_half(x) * sin.squeeze(2)
        return x_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        r"""Rotate the second half of the last dimension for RoPE: $[-x_{D/2:}, x_{:D/2}]$."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
