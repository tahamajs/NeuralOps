"""
AdaSVD: Adaptive SVD with per-layer importance allocation and iterative ALS refinement.
Registered as 'adasvd'.
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


@COMPRESSOR_REGISTRY.register("adasvd")
class AdaSVDCompressor(BaseCompressor):
    """
    Adaptive SVD Compressor (AdaSVD).

    Assigns SVD ranks dynamically across layers proportional to spectral importance
    (nuclear norm ||W||_* = sum(sigma_i)), initializes factorized matrices with
    symmetric square-root singular value distribution (A = U * sqrt(Sigma), B = sqrt(Sigma) * V^T),
    refines factors via Alternating Least Squares (ALS), and replaces dense nn.Linear layers
    with factorized SVDLinear modules.

    Args:
        model: PyTorch neural network to compress.
        rank: Optional baseline rank for all layers or global target rank.
        target_param_ratio: Target parameter retention ratio (e.g. 0.7 for 30% reduction).
        num_iters: Number of alternating least squares refinement iterations.
        target_modules: Optional list of module name substrings to target.
    """

    def __init__(
        self,
        model: nn.Module,
        rank: Optional[int] = None,
        target_param_ratio: float = 0.7,
        num_iters: int = 5,
        target_modules: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="adasvd", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            rank = getattr(cfg, "adasvd_rank", getattr(cfg, "rank", rank))
            target_param_ratio = getattr(
                cfg, "adasvd_target_ratio", getattr(cfg, "target_param_ratio", target_param_ratio)
            )
            num_iters = getattr(cfg, "adasvd_iters", getattr(cfg, "num_iters", num_iters))

        self.rank = rank
        self.target_param_ratio = float(target_param_ratio)
        self.num_iters = int(num_iters)
        self.target_modules = target_modules

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if not isinstance(module, nn.Linear):
            return False
        if isinstance(module, SVDLinear):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    @staticmethod
    def _compute_nuclear_norm(W: torch.Tensor) -> float:
        """Compute nuclear norm ||W||_* = sum(sigma_i) in FP32."""
        with torch.no_grad():
            s = torch.linalg.svdvals(W.float())
            return float(s.sum().item())

    def compress_linear(
        self,
        layer: nn.Linear,
        rank: int,
    ) -> SVDLinear:
        """
        Compress a single nn.Linear layer using symmetric SVD and ALS refinement.
        """
        out_f = layer.out_features
        in_f = layer.in_features
        r = max(1, min(rank, out_f, in_f))
        device = layer.weight.device
        dtype = layer.weight.dtype

        # Compute full SVD in FP32
        W = layer.weight.data.float()
        U, S, Vt = torch.linalg.svd(W, full_matrices=False)

        # Symmetric square-root singular value initialization
        sqrt_S = torch.sqrt(torch.clamp(S[:r], min=1e-12))
        A = U[:, :r] * sqrt_S.unsqueeze(0)       # [out_features, r]
        B = sqrt_S.unsqueeze(1) * Vt[:r, :]      # [r, in_features]

        # Alternating Least Squares (ALS) refinement
        eps_eye = 1e-6 * torch.eye(r, device=device, dtype=torch.float32)
        for _ in range(self.num_iters):
            # Update A: min ||W - A @ B||_F^2 => A = W @ B^T @ (B @ B^T + eps*I)^-1
            BBt = B @ B.t() + eps_eye
            A = W @ B.t() @ torch.linalg.inv(BBt)

            # Update B: min ||W - A @ B||_F^2 => B = (A^T @ A + eps*I)^-1 @ A^T @ W
            AtA = A.t() @ A + eps_eye
            B = torch.linalg.inv(AtA) @ A.t() @ W

        # Instantiate structural SVDLinear
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
        """
        Executes adaptive rank assignment and ALS refinement across all linear layers.
        """
        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("AdaSVD: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        # Step 1: Compute spectral nuclear norms
        layer_importances = {}
        total_importance = 0.0
        for name, module in target_layers:
            imp = self._compute_nuclear_norm(module.weight.data)
            layer_importances[name] = max(imp, 1e-8)
            total_importance += layer_importances[name]

        # Step 2: Assign ranks per layer
        for name, module in target_layers:
            max_r = min(module.in_features, module.out_features)
            if self.rank is not None:
                # Use specified rank or scale by normalized importance
                frac = layer_importances[name] / (total_importance / len(target_layers))
                allocated_rank = int(round(self.rank * frac))
            else:
                frac = layer_importances[name] / total_importance
                allocated_rank = int(round(max_r * self.target_param_ratio * frac * len(target_layers)))

            allocated_rank = max(1, min(allocated_rank, max_r))
            svd_layer = self.compress_linear(module, rank=allocated_rank)
            replace_module(self.model, name, svd_layer)

        stats = self.get_compression_stats()
        return self.model, stats


def adasvd_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> nn.Module:
    """AdaSVD factory function."""
    compressor = AdaSVDCompressor(model, comp_config=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


adasvd_compressor_factory = adasvd_factory

__all__ = ["AdaSVDCompressor", "adasvd_factory", "adasvd_compressor_factory"]
