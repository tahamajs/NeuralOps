"""
SVD-based low-rank linear layer replacement with optional Fused SVD kernel acceleration.
"""

from __future__ import annotations

from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn

from framework.attentions.fused_svd_kernel import FusedSVDLinear, fused_svd_linear


class SVDLinearLayer(nn.Module):
    """
    Replace a linear layer with U @ diag(S) @ V, keeping top-`rank` components.
    Optionally dispatches fused in-SRAM kernel to eliminate memory roundtrips.
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int,
        use_fused_kernel: bool = False,
        use_rms_norm: bool = False,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_features: int = linear_layer.in_features
        self.out_features: int = linear_layer.out_features
        self.rank: int = min(rank, self.in_features, self.out_features)
        self.use_fused_kernel: bool = use_fused_kernel
        self.use_rms_norm: bool = use_rms_norm
        device = linear_layer.weight.device
        dtype = linear_layer.weight.dtype
        with torch.no_grad():
            W = linear_layer.weight.data.float()
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
            # Symmetrical square-root singular energy distribution
            sqrt_s = torch.sqrt(torch.clamp(S[: self.rank], min=1e-12))
            U_weighted = (U[:, : self.rank] * sqrt_s.unsqueeze(0)).to(device=device, dtype=dtype)
            V_weighted = (sqrt_s.unsqueeze(1) * Vh[: self.rank, :]).to(device=device, dtype=dtype)
            S_param = S[: self.rank].to(device=device, dtype=dtype)
            Vh_param = Vh[: self.rank, :].to(device=device, dtype=dtype)

        self.U = nn.Parameter(U_weighted)
        self.V = nn.Parameter(V_weighted)
        self.S = nn.Parameter(S_param)
        self.Vh = nn.Parameter(Vh_param)

        if use_rms_norm:
            self.norm_weight = nn.Parameter(torch.ones(self.rank, dtype=dtype, device=device))
        else:
            self.register_parameter("norm_weight", None)

        if linear_layer.bias is not None:
            self.bias = nn.Parameter(linear_layer.bias.data.clone().to(device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional fused kernel acceleration."""
        # RESEARCH FIX: Fused kernel dispatch eliminates intermediate DRAM latency
        if self.use_fused_kernel:
            return fused_svd_linear(
                x=x,
                V=self.V,
                U=self.U,
                norm_weight=self.norm_weight if self.use_rms_norm else None,
                bias=self.bias,
                eps=self.eps,
                use_rms_norm=self.use_rms_norm,
            )

        # Standard forward pass: (x @ V^T) @ U^T + bias
        x_v = torch.matmul(x, self.V.t())
        out = torch.matmul(x_v, self.U.t())
        if self.bias is not None:
            out = out + self.bias
        return out


from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY


class SVDLinearCompressor(BaseCompressor):
    """Compresses model by replacing Linear layers with SVDLinearLayer."""

    def __init__(self, model: nn.Module, rank: int = 64, **kwargs) -> None:
        super().__init__(model, name="svd_linear", **kwargs)
        self.rank = rank

    def compress(self):
        """Compress linear layers with SVD factorization."""
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear):
                parent_name = name.rsplit(".", 1)[0] if "." in name else ""
                child_name = name.rsplit(".", 1)[-1] if "." in name else name
                parent = self.model.get_submodule(parent_name) if parent_name else self.model
                setattr(parent, child_name, SVDLinearLayer(module, rank=self.rank))

        self._compressed_params = self._count_params(self.model)
        return self.model, self.get_compression_stats()


@COMPRESSOR_REGISTRY.register("svd_linear")
@COMPRESSOR_REGISTRY.register("svd_linear_compressor")
def svd_linear_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """SVD Linear compressor factory."""
    rank = kwargs.pop("rank", getattr(comp_config, "rank", 64) if comp_config is not None else 64)
    compressor = SVDLinearCompressor(model, rank=rank, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "SVDLinearLayer",
    "SVDLinearCompressor",
    "svd_linear_factory",
    "FusedSVDLinear",
    "fused_svd_linear",
]
