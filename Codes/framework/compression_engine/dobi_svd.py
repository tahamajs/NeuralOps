"""
Dobi-SVD: Differentiable SVD layer with learnable spectral gating.
Registered as 'dobi'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


class DobiSVDLinearLayer(nn.Module):
    """
    Differentiable SVD Linear Layer (Dobi-SVD).

    Decomposes linear transformations into low-rank factorized parameters
    (U: [out_features, rank], V: [rank, in_features]) with a learnable continuous
    spectral gate vector alpha: [rank], allowing differentiable rank pruning and fine-tuning:
        y = (x @ (V * sigmoid(alpha)).T) @ U.T + bias
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        bias: bool = True,
        init_gate_val: float = 2.0,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = max(1, min(rank, in_features, out_features))

        self.V = nn.Parameter(torch.empty(self.rank, in_features, dtype=dtype, device=device))
        self.U = nn.Parameter(torch.empty(out_features, self.rank, dtype=dtype, device=device))
        # Learnable spectral gate parameter initialized to pass ~90% through sigmoid
        self.gate = nn.Parameter(torch.full((self.rank,), fill_value=init_gate_val, dtype=dtype, device=device))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.orthogonal_(self.U)
        nn.init.orthogonal_(self.V)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def set_low_rank_factors(self, U_factor: torch.Tensor, V_factor: torch.Tensor) -> None:
        """Initialize U and V parameters directly from SVD factors."""
        with torch.no_grad():
            self.U.copy_(U_factor)
            self.V.copy_(V_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Differentiable forward pass with spectral gating."""
        g = torch.sigmoid(self.gate).unsqueeze(1)  # [rank, 1]
        V_gated = self.V * g                       # [rank, in_features]
        latent = F.linear(x, V_gated)              # [..., rank]
        out = F.linear(latent, self.U, self.bias)  # [..., out_features]
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, bias={self.bias is not None}"


@COMPRESSOR_REGISTRY.register("dobi")
class DobiSVDCompressor(BaseCompressor):
    """
    Dobi-SVD Compressor: Differentiable SVD with learnable spectral gating.

    Extracts truncated SVD factors from pretrained nn.Linear weights with symmetric
    square-root singular value distribution (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T),
    and replaces layers with DobiSVDLinearLayer containing learnable gating parameters.

    Args:
        model: PyTorch model.
        init_rank_ratio: Fraction of max rank for initial bottleneck (0.0 < ratio <= 1.0).
        rank: Optional target rank override.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        init_rank_ratio: float = 0.5,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="dobi", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            init_rank_ratio = getattr(
                cfg, "dobi_init_rank_ratio", getattr(cfg, "init_rank_ratio", init_rank_ratio)
            )
            rank = getattr(cfg, "dobi_rank", getattr(cfg, "rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.init_rank_ratio = float(init_rank_ratio)
        self.rank = rank
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, DobiSVDLinearLayer):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Linear,
        rank: Optional[int] = None,
    ) -> DobiSVDLinearLayer:
        """Decompose a single linear layer into DobiSVDLinearLayer."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)

        if rank is not None:
            r = max(1, min(rank, max_r))
        else:
            r = max(1, min(int(round(max_r * self.init_rank_ratio)), max_r))

        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Symmetric square-root singular value distribution
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        dobi_layer = DobiSVDLinearLayer(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        dobi_layer.set_low_rank_factors(U_r, V_r)
        if layer.bias is not None:
            dobi_layer.bias.data.copy_(layer.bias.data)

        return dobi_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("Dobi-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            new_layer = self.compress_linear(module, rank=self.rank)
            replace_module(self.model, name, new_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def dobi_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Dobi factory function."""
    compressor = DobiSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
