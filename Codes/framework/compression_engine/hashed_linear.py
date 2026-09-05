"""
HashedNets: Vectorized Weight Sharing via Hash Tables with Sign Hashing and Optional LoRA Adapters.
Reference: Chen et al., 'Compressing Neural Networks with the Hashing Trick' (ICML 2015).
Production hardened for multi-architecture compatibility (nn.Linear, Conv1D, fused projections).
"""

from __future__ import annotations

import logging
import math
from typing import Any, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

_HASH_PRIME = 2147483647  # large prime for a 2-universal affine family
_HASH_A = 1103515245
_HASH_B = 12345
_SIGN_A = 1664525
_SIGN_B = 1013904223


def fast_hash_and_sign(
    indices_i: Union[int, torch.Tensor],
    indices_j: Optional[Union[int, torch.Tensor]] = None,
    n_buckets: int = 1024,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Memory-efficient vectorized hash index mapping and unbiased sign generation.
    Supports either passing integer dimensions (out_features, in_features) to generate
    broadcasting coordinates without expanding full tensors, or passing explicit 2D index tensors.
    """
    if isinstance(indices_i, int) and isinstance(indices_j, int):
        i_vec = torch.arange(indices_i, device=device, dtype=torch.int64).unsqueeze(1)
        j_vec = torch.arange(indices_j, device=device, dtype=torch.int64).unsqueeze(0)
    elif isinstance(indices_i, torch.Tensor) and isinstance(indices_j, torch.Tensor):
        i_vec = indices_i.long()
        j_vec = indices_j.long()
    else:
        raise ValueError("indices_i and indices_j must be both integers or both torch.Tensors")

    flat = (i_vec * (indices_j if isinstance(indices_j, int) else (j_vec.max() + 1)) + j_vec).long()
    mixed = (_HASH_A * flat + _HASH_B) % _HASH_PRIME
    hash_idx = (mixed % max(1, int(n_buckets))).to(torch.int32)

    sign_mix = (_SIGN_A * flat + _SIGN_B) % _HASH_PRIME
    sign_bit = (sign_mix & 1)
    sign_val = (sign_bit.to(torch.int8) * 2 - 1)
    return hash_idx, sign_val


class HashedLinear(nn.Module):
    """
    Linear layer with weight sharing via a compact parameter hash table and optional LoRA adapters.
    W_{i, j} = xi(i, j) * w[h(i, j)]
    y = x W^T + b + (x A^T B^T) * (alpha / r)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        n_buckets: int = 65536,
        bias: bool = True,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
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

        # Precompute coordinate hash and sign buffers
        hash_idx, sign_val = fast_hash_and_sign(
            out_features, in_features, self.n_buckets, device=device
        )
        self.register_buffer("hash_idx", hash_idx)
        if use_sign:
            self.register_buffer("sign_val", sign_val)
        else:
            self.register_buffer("sign_val", torch.ones_like(sign_val, dtype=torch.int8))

        # Optional LoRA adapter parameters
        if lora_rank is not None and lora_rank > 0:
            self.lora_rank = int(lora_rank)
            self.lora_alpha = float(lora_alpha)
            self.lora_scale = self.lora_alpha / self.lora_rank
            self.lora_A = nn.Parameter(
                torch.empty(self.lora_rank, in_features, dtype=dtype, device=device)
            )
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, self.lora_rank, dtype=dtype, device=device)
            )
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.lora_rank = 0
            self.lora_alpha = 1.0
            self.lora_scale = 1.0
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass reconstructing W on-the-fly, with optional LoRA residual."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        W = self.get_weight()
        out = F.linear(x_flat, W, self.bias)

        if self.lora_A is not None and self.lora_B is not None:
            lora_out = F.linear(F.linear(x_flat, self.lora_A), self.lora_B) * self.lora_scale
            out = out + lora_out

        return out.view(*orig_shape[:-1], self.out_features)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix for compatibility."""
        return self.get_weight()

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        W = self.weight_table[self.hash_idx.long()]
        if self.use_sign and self.sign_val is not None:
            W = W * self.sign_val.to(dtype=self.weight_table.dtype)
        return W

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        n_buckets: int = 65536,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
    ) -> "HashedLinear":
        """
        Create HashedLinear from existing nn.Linear or Conv1D layer using
        FP32 scatter-averaging.
        """
        cls_name = layer.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(layer, "nf")
        raw_w = layer.weight.data

        if is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t()  # [out_features, in_features]
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w

        bias = layer.bias.data if layer.bias is not None else None

        hashed = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            bias=bias is not None,
            use_sign=use_sign,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            dtype=W.dtype,
            device=W.device,
        )

        # FP32 scatter accumulation for numerical precision
        W_flat = W.float().flatten()
        hash_flat = hashed.hash_idx.flatten().long()
        sign_flat = hashed.sign_val.flatten().to(dtype=torch.float32)
        weighted_W = W_flat * sign_flat

        table_accum = torch.zeros(hashed.n_buckets, dtype=torch.float32, device=W.device)
        counts = torch.bincount(hash_flat, minlength=hashed.n_buckets).float()

        table_accum.scatter_add_(0, hash_flat, weighted_W)

        # Safe mean with zero-count handling
        non_zero = counts > 0
        table_accum[non_zero] = table_accum[non_zero] / counts[non_zero]
        bound = 1.0 / math.sqrt(max(1, in_f))
        table_accum[~non_zero] = torch.empty((~non_zero).sum(), device=W.device).uniform_(-bound, bound)

        hashed.weight_table.data.copy_(table_accum.to(dtype=W.dtype))

        if bias is not None and hashed.bias is not None:
            hashed.bias.data.copy_(bias)

        return hashed


__all__ = ["fast_hash_and_sign", "HashedLinear"]
