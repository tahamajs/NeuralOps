"""
Adaptive Transformer Compressor: automatically compresses attention and linear
layers using singular energy-threshold SVD with symmetric square-root singular value distribution.
Registered as 'adaptive_svd'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.attentions.compressed_attention import CompressedAttentionLayer
from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("adaptive_svd")
class AdaptiveTransformerCompressor(BaseCompressor):
    """
    Adaptive Transformer Compressor.

    Compresses nn.MultiheadAttention and nn.Linear layers using singular energy thresholding:
        sum_{i=1}^r sigma_i^2 / sum_j sigma_j^2 >= energy_threshold
    with symmetric square-root singular value allocation:
        U_factor = U_r * sqrt(Sigma_r), V_factor = sqrt(Sigma_r) * V_r^T.
    """

    def __init__(
        self,
        model: nn.Module,
        energy_threshold: float = 0.95,
        min_rank: int = 16,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adaptive_svd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            energy_threshold = getattr(cfg, "energy_threshold", energy_threshold)
            min_rank = getattr(cfg, "min_rank", getattr(cfg, "rank", min_rank))

        if "rank" in kwargs and kwargs["rank"] is not None:
            min_rank = kwargs["rank"]

        self.energy_threshold = float(energy_threshold)
        self.min_rank = int(min_rank)
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if isinstance(module, (SVDLinear, CompressedAttentionLayer)):
            return False
        if not isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def _compute_energy_rank(self, S: torch.Tensor, max_rank: int) -> int:
        """Find smallest rank satisfying cumulative singular energy ratio >= threshold."""
        s_squared = S ** 2
        total_energy = s_squared.sum()
        if total_energy <= 1e-12:
            return max(1, min(self.min_rank, max_rank))

        cum_energy = torch.cumsum(s_squared, dim=0) / total_energy
        idx = torch.where(cum_energy >= self.energy_threshold)[0]
        r = int(idx[0].item()) + 1 if len(idx) > 0 else max_rank
        return max(1, min(max(r, self.min_rank), max_rank))

    def compress_linear(self, layer: nn.Linear) -> SVDLinear:
        """Decompose a single linear layer via energy-thresholded SVD."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)
        device = layer.weight.device
        dtype = layer.weight.dtype

        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        r = self._compute_energy_rank(S, max_r)

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
        """Compress target attention and linear modules."""
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaptiveTransformerCompressor: No compressible layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            if isinstance(module, nn.MultiheadAttention):
                try:
                    embed_dim = module.embed_dim
                    r = max(1, min(self.min_rank, embed_dim))
                    if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
                        _, S, _ = torch.linalg.svd(module.in_proj_weight.data.float(), full_matrices=False)
                        r = self._compute_energy_rank(S, min(embed_dim, 3 * embed_dim))
                    compressed = CompressedAttentionLayer(
                        original_layer=module,
                        rank_qkv=r,
                        rank_out=r,
                    )
                    replace_module(self.model, name, compressed)
                except Exception as e:
                    logger.warning("Failed to compress attention layer %s: %s", name, e)
            elif isinstance(module, nn.Linear):
                svd_layer = self.compress_linear(module)
                replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def adaptive_svd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """Adaptive SVD factory function."""
    energy_threshold = getattr(comp_config, "energy_threshold", kwargs.pop("energy_threshold", 0.92))
    min_rank = getattr(comp_config, "min_rank", kwargs.pop("min_rank", kwargs.pop("rank", 16)))
    compressor = AdaptiveTransformerCompressor(
        model, energy_threshold=energy_threshold, min_rank=min_rank, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model


adaptive_compressor_factory = adaptive_svd_factory

__all__ = [
    "AdaptiveTransformerCompressor",
    "adaptive_svd_factory",
    "adaptive_compressor_factory",
]
