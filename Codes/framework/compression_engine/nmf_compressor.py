"""
NMF: Non-negative Matrix Factorization Linear Compressor.
Registered in COMPRESSOR_REGISTRY as 'nmf'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class NMFLinear(nn.Module):
    """
    Linear layer decomposed via Non-negative Matrix Factorization (NMF):
    W = H @ V + shift
    - H in R_+^{out_features x rank}
    - V in R_+^{rank x in_features}
    - shift in R

    Forward pass:
    y = ((x @ V^T) @ H^T) + shift * sum(x, dim=-1, keepdim=True) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        shift: float = 0.0,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)
        self.register_buffer("shift", torch.tensor(shift, dtype=dtype or torch.float32, device=device))

        self.H = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.H, 0.0, bound)
        nn.init.uniform_(self.V, 0.0, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward pass using factorized matrix multiplications.
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # Factorized forward: (x @ V^T) @ H^T
        x_v = F.linear(x_flat, self.V)  # (K, rank)
        out = F.linear(x_v, self.H)     # (K, out_features)

        # Account for constant baseline shift if non-zero
        if self.shift.abs() > 1e-8:
            x_sum = x_flat.sum(dim=-1, keepdim=True)  # (K, 1)
            out = out + x_sum * self.shift

        if self.bias is not None:
            out = out + self.bias

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return torch.matmul(self.H, self.V) + self.shift

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 64,
        max_iter: int = 100,
    ) -> "NMFLinear":
        """
        Fit Non-negative Matrix Factorization on linear layer weights using
        PyTorch Multiplicative Updates (Lee & Seung 2001).
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        min_val = W.min().item()
        shift_val = min_val - 1e-5 if min_val < 0 else 0.0
        W_pos = (W.float() - shift_val).clamp_min(1e-6)

        # Try sklearn if available on CPU for fast coordinate descent
        fitted = False
        H_init: Optional[torch.Tensor] = None
        V_init: Optional[torch.Tensor] = None

        try:
            from sklearn.decomposition import NMF as skNMF
            sk_model = skNMF(
                n_components=r,
                init="nndsvda",
                max_iter=max_iter,
                tol=1e-4,
                random_state=42,
            )
            H_np = sk_model.fit_transform(W_pos.cpu().numpy())
            V_np = sk_model.components_
            H_init = torch.tensor(H_np, dtype=W.dtype, device=W.device)
            V_init = torch.tensor(V_np, dtype=W.dtype, device=W.device)
            fitted = True
        except Exception:
            fitted = False

        if not fitted or H_init is None or V_init is None:
            # Native PyTorch Multiplicative Updates (GPU/MPS/CPU compatible)
            # NNDSVD-like initialization via SVD
            U, S, Vt = torch.linalg.svd(W_pos, full_matrices=False)
            H_init = (U[:, :r].abs() * torch.sqrt(S[:r].unsqueeze(0))).clamp_min(1e-4)
            V_init = (torch.sqrt(S[:r].unsqueeze(1)) * Vt[:r, :].abs()).clamp_min(1e-4)

            for _ in range(max_iter):
                # V update: V = V * (H^T @ W_pos) / (H^T @ H @ V + eps)
                HT = H_init.t()
                HTW = torch.matmul(HT, W_pos)
                HTHV = torch.matmul(torch.matmul(HT, H_init), V_init).clamp_min(1e-10)
                V_init = V_init * (HTW / HTHV)

                # H update: H = H * (W_pos @ V^T) / (H @ V @ V^T + eps)
                VT = V_init.t()
                WVT = torch.matmul(W_pos, VT)
                HVVT = torch.matmul(torch.matmul(H_init, V_init), VT).clamp_min(1e-10)
                H_init = H_init * (WVT / HVVT)

        nmf_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            shift=shift_val,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        nmf_layer.H.data.copy_(H_init.to(dtype=W.dtype))
        nmf_layer.V.data.copy_(V_init.to(dtype=W.dtype))

        if bias is not None and nmf_layer.bias is not None:
            nmf_layer.bias.data.copy_(bias)

        return nmf_layer


class NMFCompressor(BaseCompressor):
    """
    Non-negative Matrix Factorization (NMF) Compressor.
    Decomposes linear layer weights into low-rank non-negative factors.

    Args:
        model: Model to compress.
        rank: Target rank for NMF factorization.
        max_iter: Maximum multiplicative update iterations.
        cfg: Optional configuration object.
        target_layers: Optional layer name filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        max_iter: int = 100,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="nmf", **kwargs)
        self.rank = rank or getattr(cfg, "nmf_rank", None) or getattr(cfg, "rank", 64)
        self.max_iter = max_iter or getattr(cfg, "nmf_max_iter", 100)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress model linear layers via NMF."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, NMFLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                r = min(self.rank, max_rank)
                if r >= max_rank and max_rank > 4:
                    continue

                nmf_layer = NMFLinear.from_linear(module, rank=r, max_iter=self.max_iter)
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, nmf_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "nmf_rank": self.rank,
                "max_iter": self.max_iter,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> NMFLinear:
        """Compress a single linear layer into NMFLinear."""
        return NMFLinear.from_linear(linear, rank=self.rank, max_iter=self.max_iter)

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("nmf")
def nmf_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for NMF compression."""
    effective_rank = rank or getattr(comp_config, "nmf_rank", None) or getattr(comp_config, "rank", 64)
    max_iter = getattr(comp_config, "nmf_max_iter", 100)
    compressor = NMFCompressor(model, rank=effective_rank, max_iter=max_iter, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
