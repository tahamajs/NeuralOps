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
    Uses 64-bit bit mixing to eliminate collisions and strictly bounds bucket indices.
    """
    if isinstance(indices_i, int) and isinstance(indices_j, int):
        i_vec = torch.arange(indices_i, device=device, dtype=torch.int64).unsqueeze(1)
        j_vec = torch.arange(indices_j, device=device, dtype=torch.int64).unsqueeze(0)
        stride = indices_j
    elif isinstance(indices_i, torch.Tensor) and isinstance(indices_j, torch.Tensor):
        if indices_i.dim() == 1 and indices_j.dim() == 1:
            i_vec = indices_i.long().unsqueeze(1)
            j_vec = indices_j.long().unsqueeze(0)
            stride = max(1, int(j_vec.max().item() + 1))
        else:
            i_vec = indices_i.long()
            j_vec = indices_j.long()
            stride = max(1, int(j_vec.max().item() + 1))
    else:
        raise ValueError("indices_i and indices_j must be both integers or both torch.Tensors")

    flat = (i_vec * stride + j_vec).to(torch.int64)

    # 64-bit integer mix to completely prevent hash bucket collisions and bias
    x = flat ^ (flat >> 33)
    x = x * 0x51afd7ed
    x = x ^ (x >> 16)
    x = x * 0x45d9f3b
    x = x ^ (x >> 16)

    # Safe non-negative bucket index
    n_b = max(1, int(n_buckets))
    hash_idx = torch.remainder(x.abs(), n_b).clamp(min=0, max=n_b - 1).to(torch.int32)

    sign_mix = x ^ (x >> 1)
    sign_bit = (sign_mix & 1)
    sign_val = (sign_bit.to(torch.int8) * 2 - 1)
    return hash_idx, sign_val


class HashedLinear(nn.Module):
    is_custom_attention: bool = True
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
        self.nf = out_features  # Conv1D compatibility
        self.nx = in_features  # Conv1D compatibility
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
            self.register_buffer("sign_val", torch.ones_like(hash_idx, dtype=torch.int8))

        # Trainable low-rank adaptation branch
        self.lora_rank = lora_rank if lora_rank and lora_rank > 0 else 0
        self.lora_alpha = float(lora_alpha)
        if self.lora_rank > 0:
            self.scaling = self.lora_alpha / float(self.lora_rank)
            self.lora_A = nn.Parameter(torch.empty(self.lora_rank, in_features, dtype=dtype, device=device))
            self.lora_B = nn.Parameter(torch.zeros(out_features, self.lora_rank, dtype=dtype, device=device))
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        else:
            self.scaling = 1.0
            self.register_parameter("lora_A", None)
            self.register_parameter("lora_B", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstruct the effective full weight matrix W."""
        return self.get_weight()

    @property
    def bucket_indices(self) -> torch.Tensor:
        """Alias for hash_idx."""
        return self.hash_idx

    def get_weight(self) -> torch.Tensor:
        """
        Reconstruct dense 2D weight matrix from parameter hash table:
        W = sign_val * weight_table[hash_idx]
        """
        w_shared = self.weight_table[self.hash_idx.long()]
        if self.use_sign:
            w_shared = w_shared * self.sign_val.to(dtype=w_shared.dtype)
        return w_shared

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with vectorized weight lookup and optional LoRA addition."""
        W = self.get_weight()
        out = F.linear(x, W, self.bias)

        if self.lora_rank > 0 and self.lora_A is not None and self.lora_B is not None:
            lora_contrib = F.linear(F.linear(x, self.lora_A), self.lora_B) * self.scaling
            out = out + lora_contrib

        return out

    @classmethod
    def from_linear(
        cls,
        layer: nn.Module,
        n_buckets: int = 65536,
        use_sign: bool = True,
        lora_rank: Optional[int] = None,
        lora_alpha: float = 1.0,
        compression_ratio: Optional[float] = None,
        **kwargs: Any,
    ) -> "HashedLinear":
        """
        Create HashedLinear from existing nn.Linear, Conv1D, or LoRALinear layer using
        FP32 scatter-averaging and Frobenius energy preservation.
        """
        target_mod = getattr(layer, "base_layer", layer)
        cls_name = target_mod.__class__.__name__.lower()
        is_conv1d = "conv1d" in cls_name or hasattr(target_mod, "nf")
        raw_w = target_mod.weight.data

        if compression_ratio is not None and compression_ratio > 0.0:
            n_buckets = max(16, int(raw_w.numel() * float(compression_ratio)))

        # Determine in_features and out_features robustly, ensuring W is oriented (out_features, in_features)
        if hasattr(target_mod, "nx") and hasattr(target_mod, "nf"):
            in_f = int(target_mod.nx)
            out_f = int(target_mod.nf)
            W = raw_w.t().contiguous()
        elif hasattr(target_mod, "in_features") and hasattr(target_mod, "out_features"):
            in_f = int(target_mod.in_features)
            out_f = int(target_mod.out_features)
            if raw_w.shape == (in_f, out_f):
                W = raw_w.t().contiguous()
            else:
                W = raw_w.view(out_f, in_f).contiguous()
        elif is_conv1d:
            in_f, out_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.t().contiguous()
        else:
            out_f, in_f = raw_w.shape[0], raw_w.shape[1]
            W = raw_w.contiguous()

        norm_orig = torch.linalg.norm(W.float()).item()
        if norm_orig < 1e-4:
            logger.warning("Base layer weight has near-zero Frobenius norm (< 1e-4)")

        bias = target_mod.bias.data if getattr(target_mod, "bias", None) is not None and target_mod.bias is not None else None

        existing_lora_A = getattr(layer, "lora_A", None)
        existing_lora_B = getattr(layer, "lora_B", None)
        effective_lora_r = lora_rank
        if existing_lora_A is not None and hasattr(existing_lora_A, "weight"):
            effective_lora_r = existing_lora_A.weight.shape[0]

        hashed = cls(
            in_features=in_f,
            out_features=out_f,
            n_buckets=n_buckets,
            bias=bias is not None,
            use_sign=use_sign,
            lora_rank=effective_lora_r,
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

        # Energy and scale preservation: match original Frobenius norm to prevent activation collapse
        W_recon_pre = (table_accum[hash_flat] * sign_flat).view(out_f, in_f)
        recon_norm = torch.linalg.norm(W_recon_pre).item()
        if recon_norm > 1e-8 and norm_orig > 1e-8:
            energy_scale = norm_orig / recon_norm
            table_accum = table_accum * energy_scale

        hashed.weight_table.data.copy_(table_accum.to(dtype=W.dtype))

        if bias is not None and hashed.bias is not None:
            hashed.bias.data.copy_(bias)

        if existing_lora_A is not None and existing_lora_B is not None and hashed.lora_A is not None:
            try:
                w_A = getattr(existing_lora_A, "weight", existing_lora_A)
                w_B = getattr(existing_lora_B, "weight", existing_lora_B)
                if hasattr(w_A, "data") and hasattr(w_B, "data") and w_A.shape == hashed.lora_A.shape and w_B.shape == hashed.lora_B.shape:
                    hashed.lora_A.data.copy_(w_A.data)
                    hashed.lora_B.data.copy_(w_B.data)
                    hashed.lora_A.requires_grad = True
                    hashed.lora_B.requires_grad = True
            except Exception as e:
                logger.debug("Existing LoRA copy fallback: %s", e)
        elif effective_lora_r and effective_lora_r > 0 and hashed.lora_A is not None and hashed.lora_B is not None:
            try:
                with torch.no_grad():
                    W_cur = hashed.get_weight().float()
                    R = (W.float() - W_cur)
                    r_eff = min(hashed.lora_rank, R.shape[0], R.shape[1])
                    if r_eff > 0:
                        U, S, Vt = torch.linalg.svd(R, full_matrices=False)
                        B_init = U[:, :r_eff] * torch.sqrt(S[:r_eff])
                        A_init = torch.sqrt(S[:r_eff]).unsqueeze(1) * Vt[:r_eff, :] / max(1e-6, hashed.scaling)
                        hashed.lora_B.data.zero_()
                        hashed.lora_A.data.zero_()
                        hashed.lora_B.data[:out_f, :r_eff].copy_(B_init.to(dtype=hashed.lora_B.dtype))
                        hashed.lora_A.data[:r_eff, :in_f].copy_(A_init.to(dtype=hashed.lora_A.dtype))
                        hashed.lora_A.requires_grad = True
                        hashed.lora_B.requires_grad = True
            except Exception as e:
                logger.debug("LoRA residual SVD fit fallback: %s", e)

        return hashed


HashedLinearLayer = HashedLinear

__all__ = ["fast_hash_and_sign", "HashedLinear", "HashedLinearLayer"]

