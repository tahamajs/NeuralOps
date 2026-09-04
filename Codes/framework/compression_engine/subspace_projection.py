"""
Subspace Projection Compressor: Low-Rank Subspace Projection and Factorization.
Registered in COMPRESSOR_REGISTRY as 'subspace_projection'.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SubspaceLinear(nn.Module):
    """
    Subspace Projection Factorized Linear Layer.
    W ≈ B @ A where:
    - A in R^{rank x in_features}: orthonormal / projection basis
    - B in R^{out_features x rank}: coefficient expansion matrix

    Forward pass:
    y = ((x @ A^T) @ B^T) + bias
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

        self.A = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.B = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))

        if dtype in (torch.float16, torch.bfloat16):
            A_fp32 = torch.empty(self.rank, in_features, dtype=torch.float32)
            nn.init.orthogonal_(A_fp32)
            self.A.data.copy_(A_fp32.to(dtype=dtype, device=device))
        else:
            nn.init.orthogonal_(self.A)
        nn.init.xavier_uniform_(self.B)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @property
    def weight(self) -> torch.Tensor:
        """Reconstructed effective weight matrix (out_features, in_features)."""
        return torch.matmul(self.B, self.A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Sub-quadratic forward pass.
        y = ((x @ A^T) @ B^T) + bias
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.in_features)

        # x @ A^T: (K, rank)
        x_proj = F.linear(x_flat, self.A)
        # x_proj @ B^T + bias: (K, out_features)
        out = F.linear(x_proj, self.B, self.bias)

        return out.view(*orig_shape[:-1], self.out_features)

    def get_weight(self) -> torch.Tensor:
        """Materialize full weight matrix W = B @ A."""
        return torch.matmul(self.B, self.A)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        rank: int = 64,
        method: str = "svd",
    ) -> "SubspaceLinear":
        """
        Project linear layer weight matrix onto top-rank subspace.
        """
        W = linear.weight.data
        bias = linear.bias.data if linear.bias is not None else None
        out_f, in_f = W.shape
        r = max(1, min(rank, out_f, in_f))

        subspace_layer = cls(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=bias is not None,
            dtype=W.dtype,
            device=W.device,
        )

        U, S, Vt = torch.linalg.svd(W.float(), full_matrices=False)
        # A = V_r^T: (r, in_features)
        # B = U_r * S_r: (out_features, r)
        A_data = Vt[:r, :]
        B_data = U[:, :r] * S[:r].unsqueeze(0)

        subspace_layer.A.data.copy_(A_data.to(dtype=W.dtype))
        subspace_layer.B.data.copy_(B_data.to(dtype=W.dtype))

        if bias is not None and subspace_layer.bias is not None:
            subspace_layer.bias.data.copy_(bias)

        return subspace_layer


class SubspaceProjectionCompressor(BaseCompressor):
    """
    Subspace Projection Compression Engine.
    Compresses neural networks by projecting weight matrices onto their principal
    subspaces, replacing dense linear layers with factorized SubspaceLinear modules.

    Args:
        model: PyTorch model to compress.
        rank: Subspace rank.
        method: Subspace decomposition algorithm ('svd', 'randomized', 'pca').
        cfg: Optional configuration object.
        target_layers: Optional layer name filter.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        method: str = "svd",
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="subspace_projection", **kwargs)
        self.rank = rank or getattr(cfg, "subspace_rank", None) or getattr(cfg, "rank", 64)
        self.method = method or getattr(cfg, "subspace_method", "svd")
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers via subspace projection."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, SubspaceLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                max_rank = min(module.in_features, module.out_features)
                r = min(self.rank, max_rank)
                if r >= max_rank and max_rank > 4:
                    continue

                new_layer = SubspaceLinear.from_linear(module, rank=r, method=self.method)
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, new_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "subspace_rank": self.rank,
                "subspace_method": self.method,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> SubspaceLinear:
        """Compress a single linear layer into SubspaceLinear."""
        return SubspaceLinear.from_linear(linear, rank=self.rank, method=self.method)

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("subspace_projection")
def subspace_projection_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Subspace Projection compression."""
    effective_rank = rank or getattr(comp_config, "subspace_rank", None) or getattr(comp_config, "rank", 64)
    method = getattr(comp_config, "subspace_method", "svd")
    compressor = SubspaceProjectionCompressor(model, rank=effective_rank, method=method, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
