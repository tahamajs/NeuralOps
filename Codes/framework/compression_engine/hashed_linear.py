"""
HashedNets: Vectorized Weight Sharing via Hash Tables with Sign Hashing.
Reference: Chen et al., 'Compressing Neural Networks with the Hashing Trick' (ICML 2015).
"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

_PRIME1 = 2654435761
_PRIME2 = 2246822519
_PRIME3 = 3266489917


def fast_hash_and_sign(
    indices_i: torch.Tensor, indices_j: torch.Tensor, n_buckets: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized hash index mapping and unbiased sign generation:
    h(i, j) = ((i * PRIME1 + j * PRIME2) ^ PRIME3) % n_buckets
    xi(i, j) = 2 * (((i * PRIME2 + j * PRIME1) >> 15) & 1) - 1  in {-1, +1}
    """
    mixed = (indices_i * _PRIME1 + indices_j * _PRIME2) ^ _PRIME3
    hash_idx = (mixed % n_buckets).long()

    sign_mix = (indices_i * _PRIME2 + indices_j * _PRIME1)
    sign_bit = (sign_mix >> 15) & 1
    sign_val = (sign_bit.float() * 2.0 - 1.0)
    return hash_idx, sign_val


class HashedLinear(nn.Module):
    """
    Linear layer with weight sharing via a compact parameter hash table.
    W_{i, j} = xi(i, j) * w[h(i, j)]
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_buckets: int = 1024,
        bias: bool = True,
        use_sign: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_buckets = max(1, n_buckets)
        self.use_sign = use_sign

        # Shared compact weight table
        self.weight_table = nn.Parameter(
            torch.empty(self.n_buckets, dtype=dtype, device=device)
        )
        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.weight_table, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        # Precompute hash table indices and sign masks
        i_idx = torch.arange(out_features, device=device).unsqueeze(1).expand(-1, in_features)
        j_idx = torch.arange(in_features, device=device).unsqueeze(0).expand(out_features, -1)
        hash_idx, sign_val = fast_hash_and_sign(i_idx, j_idx, self.n_buckets)

        self.register_buffer("hash_idx", hash_idx)
        if use_sign:
            self.register_buffer("sign_val", sign_val.to(dtype=dtype or torch.float32))
        else:
            self.register_buffer("sign_val", torch.ones_like(sign_val, dtype=dtype or torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass reconstructing W on-the-fly from shared parameter table.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        W = self.weight_table[self.hash_idx] * self.sign_val
        out = F.linear(x_flat, W, self.bias)
        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return self.weight_table[self.hash_idx] * self.sign_val

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        n_buckets: int = 1024,
        use_sign: bool = True,
    ) -> "HashedLinear":
        """
        Create HashedLinear from existing nn.Linear using vectorized scatter-averaging.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape

        hashed = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            bias=bias is not None,
            use_sign=use_sign,
            dtype=W.dtype,
            device=W.device,
        )

        # Vectorized initialization using torch.scatter_add_ in O(out_f * in_f)
        W_flat = W.float().flatten()
        hash_flat = hashed.hash_idx.flatten()
        sign_flat = hashed.sign_val.flatten()
        weighted_W = W_flat * sign_flat

        table_accum = torch.zeros(hashed.n_buckets, dtype=torch.float32, device=W.device)
        counts = torch.zeros(hashed.n_buckets, dtype=torch.float32, device=W.device)

        table_accum.scatter_add_(0, hash_flat, weighted_W)
        counts.scatter_add_(0, hash_flat, torch.ones_like(W_flat))

        # Safe mean with zero-count handling
        non_zero = counts > 0
        table_accum[non_zero] = table_accum[non_zero] / counts[non_zero]
        bound = 1.0 / math.sqrt(max(1, in_f))
        table_accum[~non_zero] = torch.empty((~non_zero).sum(), device=W.device).uniform_(-bound, bound)

        hashed.weight_table.data.copy_(table_accum.to(dtype=W.dtype))

        if bias is not None and hashed.bias is not None:
            hashed.bias.data.copy_(bias)

        return hashed
