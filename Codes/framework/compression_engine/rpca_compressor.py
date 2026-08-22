"""
Robust PCA: decompose W = L + S, factorizing low-rank component L with structural SVDLinear.
Registered as 'rpca'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("rpca")
class RPCACompressor(BaseCompressor):
    """
    Robust Principal Component Analysis (RPCA) Low-Rank Compressor.

    Decomposes weight matrices W into low-rank component L and sparse outlier matrix S
    via Principal Component Pursuit / alternating soft-thresholding:
        min ||L||_* + lambda ||S||_1  s.t.  W = L + S
    Factorizes the low-rank component L into symmetric factors (U * sqrt(Sigma), sqrt(Sigma) * V^T)
    and replaces nn.Linear with factorized SVDLinear modules for physical parameter and FLOP reduction.

    Args:
        model: PyTorch model.
        rank: Target bottleneck rank r.
        mu: Soft-thresholding sparsity penalty parameter.
        max_iter: Maximum number of alternating low-rank / sparse iterations.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: int = 64,
        mu: float = 1e-3,
        max_iter: int = 5,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="rpca", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "rpca_rank", getattr(cfg, "rank", rank))
            mu = getattr(cfg, "rpca_mu", getattr(cfg, "mu", mu))
            max_iter = getattr(cfg, "rpca_max_iter", getattr(cfg, "max_iter", max_iter))

        self.rank = int(rank)
        self.mu = float(mu)
        self.max_iter = int(max_iter)
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Linear,
        rank: int,
    ) -> SVDLinear:
        """Decompose a single linear layer via RPCA and return factorized SVDLinear."""
        out_f = layer.out_features
        in_f = layer.in_features
        r = max(1, min(rank, out_f, in_f))
        device = layer.weight.device
        dtype = layer.weight.dtype

        W = layer.weight.data.float()
        L = W.clone()
        S_sparse = torch.zeros_like(W)

        # Inexact ALM / PCP Iterations in FP32
        for _ in range(self.max_iter):
            # Low-rank projection via truncated SVD
            U, S_vals, Vt = torch.linalg.svd(L, full_matrices=False)
            sqrt_S = torch.sqrt(torch.clamp(S_vals[:r], min=1e-12))
            U_r = U[:, :r] * sqrt_S.unsqueeze(0)
            V_r = sqrt_S.unsqueeze(1) * Vt[:r, :]
            L_recon = U_r @ V_r

            # Sparse soft-thresholding update on residual
            resid = W - L_recon
            S_sparse = torch.sign(resid) * torch.clamp(torch.abs(resid) - self.mu, min=0.0)

            # Update low-rank target
            L = W - S_sparse

        # Final factorization of L
        U, S_vals, Vt = torch.linalg.svd(L, full_matrices=False)
        sqrt_S = torch.sqrt(torch.clamp(S_vals[:r], min=1e-12))
        U_final = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_final = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(U_final, V_final)
        if layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model using RPCA decomposition."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("RPCA: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module, rank=self.rank)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def rpca_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """RPCA factory function."""
    compressor = RPCACompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


rpca_factory = rpca_compressor_factory

__all__ = [
    "RPCACompressor",
    "rpca_compressor_factory",
    "rpca_factory",
]
