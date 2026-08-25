"""
ERC-SVD: Error-controlled SVD with bounded Frobenius truncation error.
Registered as 'erc'.
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


@COMPRESSOR_REGISTRY.register("erc")
@COMPRESSOR_REGISTRY.register("erc_svd")
class ERCCompressor(BaseCompressor):
    """
    Error-Controlled SVD (ERC-SVD) Compressor.

    Finds the minimal rank r for each linear layer such that the relative Frobenius reconstruction
    error ||W - W_r||_F / ||W||_F is guaranteed to be <= allowed_error (i.e. singular energy retention
    sum_{i=1}^r sigma_i^2 / sum_j sigma_j^2 >= 1 - allowed_error^2). Factorizes weights symmetrically
    (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T) and replaces nn.Linear with SVDLinear modules.

    Args:
        model: PyTorch model.
        allowed_error: Relative Frobenius error tolerance epsilon in (0, 1).
        rank: Optional maximum rank ceiling.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        allowed_error: float = 0.05,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="erc", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            allowed_error = getattr(
                cfg, "erc_allowed_error", getattr(cfg, "allowed_error", allowed_error)
            )
            rank = getattr(cfg, "erc_rank", getattr(cfg, "rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.allowed_error = float(allowed_error)
        self.rank = rank
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        name_lower = name.lower()
        if any(skip in name_lower for skip in ["lm_head", "classifier", "head", "output_layer", "score"]):
            return False
        if min(module.in_features, module.out_features) < 16:
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Linear,
    ) -> SVDLinear:
        """Decompose a single linear layer via error-controlled SVD and return SVDLinear."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Cumulative energy calculation
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            r = 1
        else:
            cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
            target_retention = 1.0 - (self.allowed_error ** 2)
            idx = torch.where(cum_energy >= target_retention)[0]
            r = int(idx[0].item()) + 1 if len(idx) > 0 else max_r

        # Dynamic rank expansion up to max_r = min(d_in, d_out) strictly satisfying Frobenius error bound
        # (Artificial rank ceiling override removed)
        r = max(1, min(r, max_r))

        # Symmetrical square-root singular value distribution
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        U_r = (U[:, :r] * sqrt_S.unsqueeze(0)).to(dtype=dtype, device=device)
        V_r = (sqrt_S.unsqueeze(1) * Vt[:r, :]).to(dtype=dtype, device=device)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(U_r, V_r)
        if layer.bias is not None:
            svd_layer.bias.data.copy_(layer.bias.data)

        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress all linear layers in the model."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("ERC-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def erc_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """ERC factory function."""
    compressor = ERCCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


erc_svd_factory = erc_factory
erc_svd_compressor_factory = erc_factory

__all__ = [
    "ERCCompressor",
    "erc_factory",
    "erc_svd_factory",
    "erc_svd_compressor_factory",
]
