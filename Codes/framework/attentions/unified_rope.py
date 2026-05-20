import torch
import torch.nn as nn
from framework.core.registry import ATTENTION_REGISTRY, COMPRESSOR_REGISTRY, METHOD_REGISTRY

@ATTENTION_REGISTRY.register("compression_unified_rope")
class UnifiedRoPE(nn.Module):
    """
    Universal Rotary Position Embedding.
    Applies the same rotation to attention and SSM branches,
    guaranteeing positional consistency in hybrid architectures.
    """
    def __init__(self, dim: int, max_seq_len: int = 4096, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Inverse frequencies: 1/(base^(2i/dim))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        """Precompute cos & sin tables for positions up to seq_len."""
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)          # (seq_len, dim/2)
        emb = torch.cat((freqs, freqs), dim=-1)                    # (seq_len, dim)
        self.register_buffer("cos_cached", emb.cos())              # (seq_len, dim)
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.Tensor, **args) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            rotated tensor of same shape
        """
        seq_len = x.shape[1]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(2)   # (1, L, 1, D)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(2)

        x_rot = x * cos + self._rotate_half(x) * sin
        return x_rot

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates the second half of the last dimension."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    