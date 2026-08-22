"""
MGIE-SVD: Maximum Gradient / Information Entropy SVD rank selection.
Registered as 'mgie'.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("mgie")
class MGIECompressor(BaseCompressor):
    """
    MGIE-SVD Compressor (Maximum Information Entropy SVD).

    Analyzes each weight matrix's singular spectrum, computes normalized Shannon singular entropy:
        p_i = sigma_i / sum_j(sigma_j)
        H = -sum_i(p_i * ln(p_i))
    dynamically determines optimal truncation rank r_l proportional to H, factorizes into
    symmetric square-root matrices (U_r * sqrt(Sigma_r), sqrt(Sigma_r) * V_r^T), and replaces
    dense nn.Linear layers with structural SVDLinear modules.

    Args:
        model: PyTorch model.
        rank: Optional upper bound or target rank override.
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="mgie", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "mgie_rank", getattr(cfg, "rank", rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _compute_entropy_rank(self, S: torch.Tensor, max_rank: int) -> int:
        """Compute rank from singular entropy."""
        total_s = S.sum()
        if total_s <= 1e-12:
            return 1

        p = torch.clamp(S / total_s, min=1e-12, max=1.0)
        entropy = -torch.sum(p * torch.log(p)).item()
        max_entropy = math.log(float(max_rank)) if max_rank > 1 else 1.0

        normalized_h = entropy / max(1e-6, max_entropy)
        allocated_rank = max(1, int(round(normalized_h * max_rank)))

        if self.rank is not None:
            allocated_rank = min(allocated_rank, self.rank)

        return min(allocated_rank, max_rank)

    def compress_linear(
        self,
        layer: nn.Linear,
    ) -> SVDLinear:
        """Decompose a single linear layer via MGIE-SVD and return SVDLinear."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        r = self._compute_entropy_rank(S, max_r)

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
            logger.warning("MGIE-SVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def mgie_factory(model: nn.Module, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """MGIE factory function."""
    compressor = MGIECompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
