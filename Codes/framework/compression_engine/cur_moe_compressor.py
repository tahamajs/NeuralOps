"""
CUR-MoE Compressor: apply CUR cross-approximation to Mixture of Experts (MoE) modules and expert layers.
Registered in COMPRESSOR_REGISTRY as 'cur_moe' and 'cur_moe_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import get_dynamic_compression_rank, should_compress_linear

from .cur_linear import (
    CURLinear,
    compute_energy_scores,
    compute_leverage_scores,
    select_indices,
    tikhonov_pinv,
)

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("cur_moe")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor")
class CURMoECompressor(BaseCompressor):
    """
    Compresses Mixture of Experts (MoE) layers and dense FFN layers using CUR cross-approximation.

    Args:
        model: PyTorch model containing MoE or transformer layers.
        rank: Target rank for CUR decomposition.
        min_rank: Minimum allowable rank (default 8).
        method: 'leverage' (statistical leverage scores) or 'energy' (Frobenius energy).
        deterministic: Whether to use deterministic top-k selection (default True).
        max_error: Maximum allowed relative Frobenius error (default 0.20).
        fallback_to_original: Whether to preserve original layer if error > max_error (default True).
        max_retries: Number of adaptive rank escalations (default 2).
        cfg: Optional configuration object.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        min_rank: int = 8,
        method: str = "leverage",
        deterministic: bool = True,
        max_error: float = 0.20,
        fallback_to_original: bool = True,
        max_retries: int = 2,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur_moe", **kwargs)
        self.rank = rank or getattr(cfg, "cur_moe_rank", None) or getattr(cfg, "rank", 64)
        self.min_rank = min_rank or getattr(cfg, "min_rank", 8)
        self.method = method or getattr(cfg, "cur_moe_method", "leverage")
        self.deterministic = deterministic
        self.max_error = max_error if max_error is not None else getattr(cfg, "max_error", 0.20)
        self.fallback_to_original = (
            fallback_to_original if fallback_to_original is not None else getattr(cfg, "fallback_to_original", True)
        )
        self.max_retries = max(0, int(max_retries))
        self.cfg = cfg

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def _compress_single_layer(self, name: str, module: nn.Module) -> bool:
        """Compress a single linear or Conv1D layer with Frobenius validation and adaptive retries."""
        cls_name = module.__class__.__name__
        is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")
        raw_w = module.weight.data
        orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
        out_f, in_f = orig_w.shape
        max_rank = min(out_f, in_f)

        if max_rank < self.min_rank:
            return False

        # Parameter reduction check: r*(in_f + out_f + r) < in_f * out_f
        target_r = min(self.rank, max_rank)
        while target_r >= self.min_rank and (target_r * (in_f + out_f + target_r)) >= 0.85 * (in_f * out_f):
            target_r = int(target_r * 0.8)

        if target_r < self.min_rank:
            return False

        cur_r = target_r
        best_cur_layer = None
        best_err = float("inf")
        best_r = cur_r
        norm_orig = torch.norm(orig_w, p="fro").clamp_min(1e-8)

        for _ in range(self.max_retries + 1):
            try:
                candidate = CURLinear.from_linear(
                    module,
                    rank=cur_r,
                    method=self.method,
                    deterministic=self.deterministic,
                )
                recon_w = candidate.get_weight().float()
                err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()

                if err < best_err:
                    best_err = err
                    best_cur_layer = candidate
                    best_r = cur_r

                if err <= self.max_error:
                    break

                next_r = min(int(cur_r * 1.5), max_rank - 1)
                if next_r <= cur_r or (next_r * (in_f + out_f + next_r)) >= 0.85 * (in_f * out_f):
                    break
                cur_r = next_r
            except Exception as ex:
                logger.warning("CURMoECompressor: Error on %s with rank %d: %s", name, cur_r, ex)
                break

        if best_err > self.max_error:
            logger.warning(
                "CURMoECompressor: Layer %s error %.4f exceeded max_error %.4f",
                name, best_err, self.max_error
            )
            if self.fallback_to_original:
                self.fallback_layers.append(name)
                return False

        if best_cur_layer is not None:
            replace_module(self.model, name, best_cur_layer)
            self.compressed_layers.append(name)
            self.layer_errors[name] = float(best_err)
            self.layer_ranks[name] = int(best_r)
            return True

        return False

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress expert linear layers in MoE architecture, or fallback to FFN layers."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}

        # Scan for explicit MoE modules (module.experts or expert submodules)
        for name, module in list(self.model.named_modules()):
            name_lower = name.lower()
            if any(k in name_lower for k in ["router", "gate", "classifier", "score", "lm_head", "pooler", "embedding"]):
                # Protect routers, gates, and classification heads
                continue

            if hasattr(module, "experts"):
                experts_container = module.experts
                if isinstance(experts_container, (nn.ModuleList, list)):
                    for exp_idx, expert in enumerate(experts_container):
                        for child_name, child_mod in list(expert.named_children()):
                            child_full_name = f"{name}.experts.{exp_idx}.{child_name}" if name else f"experts.{exp_idx}.{child_name}"
                            is_linear_like = (
                                isinstance(child_mod, nn.Linear)
                                or "conv1d" in child_mod.__class__.__name__.lower()
                                or hasattr(child_mod, "nf")
                            )
                            if is_linear_like and not isinstance(child_mod, CURLinear):
                                self._compress_single_layer(child_full_name, child_mod)
            elif "expert" in name_lower:
                is_linear_like = (
                    isinstance(module, nn.Linear)
                    or "conv1d" in module.__class__.__name__.lower()
                    or hasattr(module, "nf")
                )
                if is_linear_like and not isinstance(module, CURLinear):
                    self._compress_single_layer(name, module)

        # Fallback to FFN linear layers if no explicit MoE structure was found
        _ATTN_NAMES = frozenset({
            "query", "key", "value",
            "q_lin", "k_lin", "v_lin",
            "q_proj", "k_proj", "v_proj",
            "out_lin", "out_proj", "o_proj",
            "c_attn",
        })
        if len(self.compressed_layers) == 0:
            for name, module in list(self.model.named_modules()):
                if not should_compress_linear(self.model, name):
                    continue
                last_part = name.split(".")[-1]
                if last_part in _ATTN_NAMES:
                    continue  # preserve pretrained attention weights intact
                name_lower = name.lower()
                if any(skip in name_lower for skip in ["classifier", "score", "lm_head", "pooler", "embedding", "router", "gate"]):
                    continue

                is_linear_like = (
                    isinstance(module, nn.Linear)
                    or "conv1d" in module.__class__.__name__.lower()
                    or hasattr(module, "nf")
                )
                if is_linear_like and not isinstance(module, CURLinear):
                    self._compress_single_layer(name, module)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        avg_err = (sum(self.layer_errors.values()) / len(self.layer_errors)) if self.layer_errors else 0.0
        stats.update(
            {
                "compressed_layers": self.compressed_layers,
                "compressed_layer_count": len(self.compressed_layers),
                "fallback_layers": self.fallback_layers,
                "fallback_layer_count": len(self.fallback_layers),
                "layer_errors": self.layer_errors,
                "layer_ranks": self.layer_ranks,
                "avg_reconstruction_error": avg_err,
                "max_error": self.max_error,
                "fallback_to_original": self.fallback_to_original,
                "cur_method": self.method,
                "cur_moe_rank": self.rank,
                "min_rank": self.min_rank,
            }
        )
        return self.model, stats


@COMPRESSOR_REGISTRY.register("cur_moe_factory")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor_factory")
def cur_moe_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR-MoE compressor."""
    effective_rank = kwargs.pop("rank", rank or getattr(comp_config, "cur_moe_rank", None) or getattr(comp_config, "rank", 64))
    min_rank = kwargs.pop("min_rank", getattr(comp_config, "min_rank", 8))
    method = kwargs.pop("method", getattr(comp_config, "cur_moe_method", "leverage"))
    deterministic = kwargs.pop("deterministic", getattr(comp_config, "deterministic", True))
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True))

    compressor = CURMoECompressor(
        model,
        rank=effective_rank,
        min_rank=min_rank,
        method=method,
        deterministic=deterministic,
        max_error=max_error,
        fallback_to_original=fallback,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
