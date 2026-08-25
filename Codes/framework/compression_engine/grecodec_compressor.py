"""
GreCoDeC: Greedy Coordinate Descent Low-Rank Compressor.
Registered in COMPRESSOR_REGISTRY as 'grecodec'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY


class GreCoDeCLinear(nn.Module):
    """
    Greedy Coordinate Descent Factorized Linear Layer.
    W ≈ U @ V
    - U in R^{out_features x rank}
    - V in R^{rank x in_features}

    Forward pass:
    y = ((x @ V^T) @ U^T) + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 16,
        bias: bool = True,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))

        bound = 1.0 / math.sqrt(max(1, in_features))
        nn.init.uniform_(self.U, -bound, bound)
        nn.init.uniform_(self.V, -bound, bound)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward execution.
        y = ((x @ V^T) @ U^T) + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        x_v = F.linear(x_flat, self.V)
        out = F.linear(x_v, self.U, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full reconstructed weight matrix W = U @ V."""
        return torch.matmul(self.U, self.V)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 16,
        max_iter: int = 15,
    ) -> "GreCoDeCLinear":
        """
        Decompose linear layer weight matrix via Greedy Coordinate Descent (GreCoDeC)
        with 15 ALS iterations and Gram-Schmidt re-orthogonalization, with SVD fallback.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        grecodec_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        R = W.float().clone()
        U_factors = torch.zeros(out_f, r, dtype=torch.float32, device=W.device)
        V_factors = torch.zeros(r, in_f, dtype=torch.float32, device=W.device)

        success = True
        try:
            for k in range(r):
                # 1. Initialize v_k from row of R with highest residual norm
                row_norms = (R ** 2).sum(dim=1)
                best_row = torch.argmax(row_norms).item()
                v_k = R[best_row, :].clone()
                v_norm = v_k.norm().clamp_min(1e-8)
                v_k = (v_k / v_norm).unsqueeze(1)  # (in_f, 1)

                # 2. ALS coordinate descent with Gram-Schmidt re-orthogonalization
                u_k = torch.zeros(out_f, 1, dtype=torch.float32, device=W.device)
                for _ in range(max(1, max_iter)):
                    # u = R @ v / ||v||^2
                    v_sq = (v_k ** 2).sum().clamp_min(1e-10)
                    u_k = torch.matmul(R, v_k) / v_sq

                    # Gram-Schmidt re-orthogonalization on u_k against prior columns U_factors[:, :k]
                    if k > 0:
                        prior_U = U_factors[:, :k]  # (out_f, k)
                        u_proj = torch.matmul(prior_U, torch.matmul(prior_U.t(), u_k) / (prior_U.norm(dim=0, keepdim=True) ** 2).clamp_min(1e-10).t())
                        u_k = u_k - u_proj

                    u_sq = (u_k ** 2).sum().clamp_min(1e-10)
                    # v = R^T @ u / ||u||^2
                    v_k = torch.matmul(R.t(), u_k) / u_sq

                    # Gram-Schmidt re-orthogonalization on v_k against prior rows V_factors[:k, :]
                    if k > 0:
                        prior_V = V_factors[:k, :].t()  # (in_f, k)
                        v_proj = torch.matmul(prior_V, torch.matmul(prior_V.t(), v_k) / (prior_V.norm(dim=0, keepdim=True) ** 2).clamp_min(1e-10).t())
                        v_k = v_k - v_proj

                # Check for degenerate vector
                if u_k.norm() < 1e-8 or v_k.norm() < 1e-8 or torch.isnan(u_k).any() or torch.isnan(v_k).any():
                    success = False
                    break

                # 3. Deflate residual: R = R - u @ v^T
                R = R - torch.matmul(u_k, v_k.t())

                U_factors[:, k] = u_k.squeeze(1)
                V_factors[k, :] = v_k.squeeze(1)

        except Exception:
            success = False

        if not success or torch.isnan(U_factors).any() or torch.isnan(V_factors).any():
            # Robust SVD fallback to guarantee orthogonal singular components
            U_svd, S_svd, Vh_svd = torch.linalg.svd(W.float(), full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S_svd[:r], min=1e-12))
            U_factors = U_svd[:, :r] * sqrt_S.unsqueeze(0)
            V_factors = sqrt_S.unsqueeze(1) * Vh_svd[:r, :]

        grecodec_layer.U.data.copy_(U_factors.to(dtype=W.dtype))
        grecodec_layer.V.data.copy_(V_factors.to(dtype=W.dtype))

        if bias is not None and grecodec_layer.bias is not None:
            grecodec_layer.bias.data.copy_(bias)

        return grecodec_layer


class GreCoDeCCompressor(BaseCompressor):
    """
    GreCoDeC (Greedy Coordinate Descent) Compressor.
    Decomposes linear layers into low-rank factors using fast greedy rank-1 updates.

    Args:
        model: PyTorch model.
        rank: Target rank for decomposition.
        max_iter: Coordinate descent iterations per rank component (default 15).
        cfg: Optional configuration object.
        target_layers: Optional target layer filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 16,
        max_iter: int = 15,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="grecodec", **kwargs)
        self.rank = rank or getattr(cfg, "grecodec_rank", None) or getattr(cfg, "rank", 16)
        self.max_iter = max_iter or getattr(cfg, "grecodec_max_iter", 15)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using GreCoDeC."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, GreCoDeCLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                r = min(self.rank, max_rank)
                if r >= max_rank and max_rank > 4:
                    continue

                grecodec_layer = GreCoDeCLinear.from_linear(
                    module, rank=r, max_iter=self.max_iter
                )
                replace_module(self.model, name, grecodec_layer)
                self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "grecodec_rank": self.rank,
                "grecodec_max_iter": self.max_iter,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> GreCoDeCLinear:
        """Compress a single linear layer into GreCoDeCLinear."""
        return GreCoDeCLinear.from_linear(linear, rank=self.rank, max_iter=self.max_iter)


@COMPRESSOR_REGISTRY.register("grecodec")
def grecodec_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for GreCoDeC compressor."""
    effective_rank = rank or getattr(comp_config, "grecodec_rank", None) or getattr(comp_config, "rank", 16)
    max_iter = getattr(comp_config, "grecodec_max_iter", 15)
    compressor = GreCoDeCCompressor(model, rank=effective_rank, max_iter=max_iter, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model

