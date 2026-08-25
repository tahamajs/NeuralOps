"""
Direct SVD Compressor: Truncated SVD low-rank factorized layer replacement.
Registered as 'direct_svd'.
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


@COMPRESSOR_REGISTRY.register("direct_svd")
class DirectSVDCompressor(BaseCompressor):
    """
    Direct SVD Compressor.

    Performs direct truncated SVD on all linear layers:
        W ≈ U_r * Sigma_r * V_r^T
    Allocates singular energy symmetrically:
        U_factor = U_r * sqrt(Sigma_r)
        V_factor = sqrt(Sigma_r) * V_r^T
    and replaces dense nn.Linear layers with factorized SVDLinear modules
    to achieve real parameter reduction and FLOP savings.

    Args:
        model: PyTorch model to compress.
        rank: Target rank r for low-rank factorization.
        rank_ratio: Optional fraction of maximum rank to retain (0.0 < rank_ratio <= 1.0).
        target_modules: Optional list of layer name filters.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = 32,
        min_rank: int = 16,
        rank_ratio: Optional[float] = None,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="direct_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "direct_svd_rank", getattr(cfg, "svd_rank", getattr(cfg, "rank", rank)))
            min_rank = getattr(cfg, "min_rank", min_rank)
            rank_ratio = getattr(cfg, "direct_svd_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        self.min_rank = max(16, int(min_rank))
        self.rank_ratio = float(rank_ratio) if rank_ratio is not None else None
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        # Exclude classifier and language model prediction heads
        name_lower = name.lower()
        skip_keywords = ["lm_head", "classifier", "head", "output_layer", "score", "embed_out", "logits_processor"]
        if any(skip in name_lower for skip in skip_keywords):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Linear,
        rank: Optional[int] = None,
    ) -> SVDLinear:
        """Decompose a single linear layer via direct SVD with proportional rank scaling and min rank floor."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)
        r_floor = max(1, min(self.min_rank, max_r))

        target_r = rank if rank is not None else self.rank
        if self.rank_ratio is not None:
            r = int(round(max_r * self.rank_ratio))
        elif target_r is not None:
            # Proportional rank scaling relative to reference hidden dimension 768
            r = int(round(target_r * (max_r / 768.0)))
        else:
            r = int(round(32 * (max_r / 768.0)))

        # Enforce minimum rank floor r_min >= 16 and physical bounds
        r = max(r_floor, min(r, max_r))

        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32 for numerical stability
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

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
            logger.warning("DirectSVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module, rank=self.rank)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def direct_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Direct SVD factory function."""
    compressor = DirectSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


direct_svd_compressor_factory = direct_svd_factory

__all__ = ["DirectSVDCompressor", "direct_svd_factory", "direct_svd_compressor_factory"]
