"""
CR (Column-Row / Interpolative Decomposition) Compressor.
Decomposes W ≈ C @ P, where C are actual columns of W.
Registered in COMPRESSOR_REGISTRY as 'cr'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class CRLinear(nn.Module):
    """
    Interpolative Decomposition / Column-Row Factorized Linear Layer.
    W ≈ C @ P where:
    - C in R^{out_features x rank}: column skeleton from W
    - P in R^{rank x in_features}: interpolative projection matrix

    Forward pass:
    y = ((x @ P^T) @ C^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 64,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = min(rank, in_features, out_features)

        self.C = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.P = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.C, -bound, bound)
        nn.init.uniform_(self.P, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution:
        y = ((x @ P^T) @ C^T) + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_p = F.linear(x_flat, self.P)
        out = F.linear(x_p, self.C, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix."""
        return torch.matmul(self.C, self.P)

    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 64) -> "CRLinear":
        """
        Perform Column Interpolative Decomposition on linear layer weight matrix.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        cr_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        W_float = W.float()
        decomposed = False

        # Attempt scipy interpolative decomposition if available
        try:
            from scipy.linalg import interpolative
            W_np = W_float.cpu().numpy()
            idx, proj = interpolative.interp_decomp(W_np, r)
            # Reconstruct full P from interpolative format
            P_full_np = interpolative.reconstruct_interp_matrix(idx, proj)  # (r, in_f)
            C_np = W_np[:, idx[:r]]  # (out_f, r)

            C_data = torch.tensor(C_np, dtype=torch.float32, device=W.device)
            P_data = torch.tensor(P_full_np, dtype=torch.float32, device=W.device)
            decomposed = True
        except Exception:
            decomposed = False

        if not decomposed:
            # PyTorch Column-Pivoted / SVD Fallback for ID
            # Select top-r energy columns as C, then P = C^+ @ W
            col_norms = (W_float ** 2).sum(dim=0)
            _, top_cols = torch.topk(col_norms, k=r, largest=True, sorted=True)
            C_data = W_float[:, top_cols]  # (out_f, r)
            C_pinv = torch.linalg.pinv(C_data)  # (r, out_f)
            P_data = torch.matmul(C_pinv, W_float)  # (r, in_f)

        cr_layer.C.data.copy_(C_data.to(dtype=W.dtype))
        cr_layer.P.data.copy_(P_data.to(dtype=W.dtype))

        if bias is not None and cr_layer.bias is not None:
            cr_layer.bias.data.copy_(bias)

        return cr_layer


class CRCompressor(BaseCompressor):
    """
    CR (Column-Row / Interpolative Decomposition) Compressor.
    Decomposes linear layers into skeleton column bases and interpolative projections.

    Args:
        model: PyTorch model.
        rank: Rank parameter (number of skeleton columns).
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cr", **kwargs)
        self.rank = rank or getattr(cfg, "cr_rank", None) or getattr(cfg, "rank", 64)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using interpolative decomposition."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, CRLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                r = min(self.rank, max_rank)
                if r >= max_rank and max_rank > 4:
                    continue

                cr_layer = CRLinear.from_linear(module, rank=r)
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, cr_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "cr_rank": self.rank,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> CRLinear:
        """Compress a single linear layer into CRLinear."""
        return CRLinear.from_linear(linear, rank=self.rank)

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("cr")
def cr_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CR compressor."""
    effective_rank = rank or getattr(comp_config, "cr_rank", None) or getattr(comp_config, "rank", 64)
    compressor = CRCompressor(model, rank=effective_rank, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
