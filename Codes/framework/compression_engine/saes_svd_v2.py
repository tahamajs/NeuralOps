"""
SAES-SVD v2: Spectrum-Aware Error Suppression SVD with iterative residual compensation.
Registered as 'saes_svd_v2' and 'saes'.
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


@COMPRESSOR_REGISTRY.register("saes_svd_v2")
@COMPRESSOR_REGISTRY.register("saes")
class SAESSVDCompressorV2(BaseCompressor):
    """Spectrum-Aware Error Suppression SVD (SAES-SVD v2)."""


    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        rank_ratio: float = 0.7,
        correcting_steps: int = 3,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="saes_svd_v2", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "saes_rank", getattr(cfg, "rank", rank))
            rank_ratio = getattr(cfg, "saes_rank_ratio", getattr(cfg, "rank_ratio", rank_ratio))
            correcting_steps = getattr(
                cfg, "saes_correcting_steps", getattr(cfg, "correcting_steps", correcting_steps)
            )

        if "rank" in kwargs and kwargs["rank"] is not None:
            rank = kwargs["rank"]

        self.rank = rank
        self.rank_ratio = float(rank_ratio)
        self.correcting_steps = int(correcting_steps)
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
        rank: Optional[int] = None,
    ) -> SVDLinear:
        """Decompose a single linear layer with spectrum-aware error suppression."""
        out_f = layer.out_features
        in_f = layer.in_features
        max_r = min(out_f, in_f)

        if rank is not None:
            r = max(1, min(rank, max_r))
        else:
            r = max(1, min(int(round(max_r * self.rank_ratio)), max_r))

        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Initial symmetric square-root singular value allocation
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A = U[:, :r] * sqrt_S.unsqueeze(0)      # [out_features, r]
        B = sqrt_S.unsqueeze(1) * Vt[:r, :]     # [r, in_features]

        # Iterative residual error suppression
        eps_eye = 1e-6 * torch.eye(r, device=device, dtype=torch.float32)
        for _ in range(self.correcting_steps):
            E = W - A @ B
            # Update A towards residual gradient
            BBt = B @ B.t() + eps_eye
            A = A + E @ B.t() @ torch.linalg.inv(BBt)

            # Update B towards residual gradient
            AtA = A.t() @ A + eps_eye
            B = B + torch.linalg.inv(AtA) @ A.t() @ (W - A @ B)

        svd_layer = SVDLinear(
            in_features=in_f,
            out_features=out_f,
            rank=r,
            bias=layer.bias is not None,
            dtype=dtype,
            device=device,
        )
        svd_layer.set_low_rank_factors(
            U_factor=A.to(dtype=dtype),
            V_factor=B.to(dtype=dtype),
        )
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
            logger.warning("SAES-SVD v2: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            svd_layer = self.compress_linear(module, rank=self.rank)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def saes_svd_v2_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """SAES-SVD v2 factory function."""
    compressor = SAESSVDCompressorV2(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


saes_factory = saes_svd_v2_factory
saes_svd_factory = saes_svd_v2_factory
saes_svd_compressor_factory = saes_svd_v2_factory

# Backward compatibility alias
SAESCompressor = SAESSVDCompressorV2

__all__ = [
    "SAESSVDCompressorV2",
    "SAESCompressor",
    "saes_svd_v2_factory",
    "saes_factory",
    "saes_svd_factory",
    "saes_svd_compressor_factory",
]

