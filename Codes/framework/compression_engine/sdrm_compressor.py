"""
SDRM: Scaled Dimension Rank Modulation.
Assigns layer rank proportional to parameter budget for edge inference and factorizes
into low-rank SVDLinear layers to achieve physical parameter reduction.
Registered as 'sdrm'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from framework.compression_engine.aa_svd_compressor import SVDLinear
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

logger = logging.getLogger(__name__)


class SDRMCompressor(BaseCompressor):
    """
    SDRM: Scaled Dimension Rank Modulation.
    Assigns layer rank proportional to parameter budget for edge inference and factorizes
    target layers into low-rank SVDLinear modules.
    """

    def __init__(
        self,
        model: nn.Module,
        total_param_ratio: float = 0.7,
        target_modules: Optional[List[str]] = None,
        max_error: float = 0.5,
        fallback_to_original: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(model, name="sdrm", **kwargs)
        cfg = kwargs.get("comp_config", kwargs.get("cfg", None))
        if cfg is not None:
            total_param_ratio = getattr(cfg, "total_param_ratio", getattr(cfg, "ratio", total_param_ratio))
            max_error = getattr(cfg, "max_error", max_error)
            fallback_to_original = getattr(cfg, "fallback_to_original", fallback_to_original)

        self.total_param_ratio = max(0.05, min(float(total_param_ratio), 1.0))
        self.target_modules = target_modules
        self.max_error = float(max_error)
        self.fallback_to_original = bool(fallback_to_original)
        self.layer_errors: Dict[str, float] = {}
        self.replaced_layers: List[Dict[str, Any]] = []

    def _is_target_module(self, name: str, module: nn.Module) -> bool:
        if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler", "embedding"]):
            return False
        cls_name = module.__class__.__name__
        is_linear_like = isinstance(module, nn.Linear) or "conv1d" in cls_name.lower()
        if not is_linear_like:
            return False
        if isinstance(module, SVDLinear):
            return False
        if not should_compress_linear(self.model, name):
            return False
        if self.target_modules is None:
            return True
        return any(t in name for t in self.target_modules)

    def compress_linear(
        self,
        layer: nn.Module,
        rank: int,
    ) -> Optional[SVDLinear]:
        """Decompose a single linear or Conv1D layer into SVDLinear."""
        raw_w = layer.weight.data
        cls_name = layer.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or (
            raw_w.ndim == 2
            and hasattr(layer, "in_features")
            and layer.in_features == raw_w.shape[0]
            and raw_w.shape[0] < raw_w.shape[1]
        )

        if is_conv1d:
            W = raw_w.t().float()
            out_f = W.shape[0]
            in_f = W.shape[1]
        else:
            W = raw_w.float()
            out_f = getattr(layer, "out_features", W.shape[0])
            in_f = getattr(layer, "in_features", W.shape[1])

        max_r = min(out_f, in_f)
        param_cap = max(1, (in_f * out_f) // (in_f + out_f) - 1)
        r = max(1, min(rank, max_r, param_cap))

        device = layer.weight.device
        dtype = layer.weight.dtype

        try:
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)
        except Exception as e:
            logger.warning("SDRM SVD failure: %s", e)
            return None

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

        # Frobenius reconstruction error check
        W_rec = torch.matmul(U_r.float(), V_r.float())
        norm_orig = torch.norm(W, p="fro")
        rel_err = (torch.norm(W - W_rec, p="fro") / (norm_orig + 1e-8)).item()

        if self.fallback_to_original and rel_err > self.max_error:
            logger.warning(
                "SDRM reconstruction error %.4f exceeded threshold %.4f. Keeping original.",
                rel_err,
                self.max_error,
            )
            return None

        svd_layer.reconstruction_error = rel_err
        return svd_layer

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress linear layers using budget-proportional rank assignment."""
        self.layer_errors = {}
        self.replaced_layers = []

        target_layers = [
            (name, mod)
            for name, mod in self.model.named_modules()
            if self._is_target_module(name, mod)
        ]

        if not target_layers:
            logger.warning("SDRM: No compressible linear layers found.")
            return self.model, self.get_compression_stats()

        for name, module in target_layers:
            raw_w = module.weight.data
            cls_name = module.__class__.__name__
            is_conv1d = "conv1d" in cls_name.lower() or (
                raw_w.ndim == 2
                and hasattr(module, "in_features")
                and module.in_features == raw_w.shape[0]
                and raw_w.shape[0] < raw_w.shape[1]
            )
            out_f = raw_w.shape[1] if is_conv1d else raw_w.shape[0]
            in_f = raw_w.shape[0] if is_conv1d else raw_w.shape[1]

            params = in_f * out_f
            allowed = params * self.total_param_ratio
            rank = max(1, int(allowed / max(in_f + out_f, 1)))

            svd_layer = self.compress_linear(module, rank=rank)
            if svd_layer is not None:
                err = getattr(svd_layer, "reconstruction_error", 0.0)
                self.layer_errors[name] = err
                self.replaced_layers.append({"name": name, "rank": svd_layer.rank, "error": err})
                replace_module(self.model, name, svd_layer)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["total_param_ratio"] = self.total_param_ratio
        stats["layer_errors"] = self.layer_errors
        stats["compressed_layers"] = len(self.replaced_layers)
        return self.model, stats


@COMPRESSOR_REGISTRY.register("sdrm")
@COMPRESSOR_REGISTRY.register("sdrm_compressor")
def sdrm_compressor_factory(model: Any, comp_config: Any = None, **kwargs: Any) -> Any:
    """SDRM compressor factory."""
    total_param_ratio = kwargs.pop(
        "total_param_ratio",
        kwargs.pop("ratio", getattr(comp_config, "total_param_ratio", 0.7) if comp_config is not None else 0.7),
    )
    compressor = SDRMCompressor(model, total_param_ratio=total_param_ratio, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model


__all__ = [
    "SDRMCompressor",
    "sdrm_compressor_factory",
]
