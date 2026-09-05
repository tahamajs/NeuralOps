"""
CUR Compressor: replaces Linear and Conv1D layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur' and 'cur_compressor'.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.model_utils import replace_module
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

from .cur_linear import CURLinear

logger = logging.getLogger(__name__)


@COMPRESSOR_REGISTRY.register("cur_compressor")
class CURCompressor(BaseCompressor):
    """
    Compressor that replaces eligible Linear and Conv1D layers with CURLinear.

    Args:
        model: PyTorch model.
        rank: Target rank for skeleton column and row selection (default 64).
        min_rank: Minimum allowable rank for compression (default 8).
        method: Selection method ('leverage' or 'energy', default 'leverage').
        deterministic: Deterministic top-k selection if True (default True).
        max_error: Maximum relative Frobenius reconstruction error allowed (default 0.20).
        fallback_to_original: If True, retains original layer when reconstruction error exceeds max_error (default True).
        max_retries: Number of adaptive rank retries when error threshold is exceeded (default 2).
        lora_r: LoRA adapter rank if adding fine-tuning adapters (default 0).
        lora_alpha: LoRA scaling factor (default 1.0).
        freeze_skeletons: If True, freeze C and R parameters during post-compression training (default False).
        target_names: Optional list of layer name substrings to target.
        exclude_names: Optional list of layer name substrings to exclude.
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
        lora_r: int = 0,
        lora_alpha: float = 1.0,
        freeze_skeletons: bool = False,
        target_names: Optional[List[str]] = None,
        exclude_names: Optional[List[str]] = None,
        cfg: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur", **kwargs)
        self.rank = rank or getattr(cfg, "cur_rank", None) or getattr(cfg, "rank", 64)
        self.min_rank = min_rank or getattr(cfg, "min_rank", 8)
        self.method = method or getattr(cfg, "cur_method", "leverage")
        self.deterministic = deterministic
        self.max_error = max_error if max_error is not None else getattr(cfg, "max_error", 0.20)
        self.fallback_to_original = (
            fallback_to_original if fallback_to_original is not None else getattr(cfg, "fallback_to_original", True)
        )
        self.max_retries = max(0, int(max_retries))
        self.lora_r = int(lora_r or getattr(cfg, "lora_r", 0))
        self.lora_alpha = float(lora_alpha or getattr(cfg, "lora_alpha", 1.0))
        self.freeze_skeletons = bool(freeze_skeletons or getattr(cfg, "freeze_skeletons", False))
        self.cfg = cfg
        self.target_names = target_names or getattr(cfg, "cur_target_names", None) or getattr(cfg, "target_layers", None)
        self.exclude_names = exclude_names or getattr(cfg, "cur_exclude_names", None) or [
            "lm_head", "classifier", "head", "score", "pooler", "embedding", "router", "gate"
        ]

        self.compressed_layers: List[str] = []
        self.fallback_layers: List[str] = []
        self.layer_errors: Dict[str, float] = {}
        self.layer_ranks: Dict[str, int] = {}

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace eligible Linear and Conv1D layers with CURLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        self.fallback_layers = []
        self.layer_errors = {}
        self.layer_ranks = {}
        visited_weight_ids: Set[int] = set()

        for name, module in list(self.model.named_modules()):
            cls_name = module.__class__.__name__
            is_linear_like = (
                isinstance(module, nn.Linear)
                or "conv1d" in cls_name.lower()
                or hasattr(module, "nf")
            )
            if not is_linear_like or isinstance(module, CURLinear):
                continue

            name_lower = name.lower()
            if any(skip in name_lower for skip in self.exclude_names):
                continue
            if not should_compress_linear(self.model, name):
                continue
            if self.target_names and not any(t in name for t in self.target_names):
                continue

            # Check tied weights
            weight_id = id(module.weight)
            if weight_id in visited_weight_ids:
                logger.info("CURCompressor: Skipping tied weight layer %s", name)
                continue

            is_conv1d = "conv1d" in cls_name.lower() or hasattr(module, "nf")
            raw_w = module.weight.data
            orig_w = raw_w.t().float() if is_conv1d else raw_w.float()
            out_f, in_f = orig_w.shape
            max_rank = min(out_f, in_f)

            if max_rank < self.min_rank:
                continue

            # Parameter reduction check: r*(in_f + out_f + r) < in_f * out_f
            target_r = min(self.rank, max_rank)
            while target_r >= self.min_rank and (target_r * (in_f + out_f + target_r)) >= 0.85 * (in_f * out_f):
                target_r = int(target_r * 0.8)

            if target_r < self.min_rank:
                logger.debug("CURCompressor: Skipping %s as target rank %d violates parameter reduction constraint", name, target_r)
                continue

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
                        lora_r=self.lora_r,
                        lora_alpha=self.lora_alpha,
                        freeze_skeletons=self.freeze_skeletons,
                    )
                    recon_w = candidate.get_weight().float()
                    err = (torch.norm(orig_w - recon_w, p="fro") / norm_orig).item()

                    if err < best_err:
                        best_err = err
                        best_cur_layer = candidate
                        best_r = cur_r

                    if err <= self.max_error:
                        break

                    # Adaptive retry: increase rank if parameter budget permits
                    next_r = min(int(cur_r * 1.5), max_rank - 1)
                    if next_r <= cur_r or (next_r * (in_f + out_f + next_r)) >= 0.85 * (in_f * out_f):
                        break
                    cur_r = next_r
                except Exception as ex:
                    logger.warning("CURCompressor: Failed CUR decomposition on %s at rank %d: %s", name, cur_r, ex)
                    break

            if best_err > self.max_error:
                logger.warning(
                    "CURCompressor: Layer %s error %.4f exceeded max_error %.4f",
                    name, best_err, self.max_error
                )
                if self.fallback_to_original:
                    self.fallback_layers.append(name)
                    continue

            if best_cur_layer is not None:
                replace_module(self.model, name, best_cur_layer)
                self.compressed_layers.append(name)
                self.layer_errors[name] = float(best_err)
                self.layer_ranks[name] = int(best_r)
                visited_weight_ids.add(weight_id)

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
                "cur_rank": self.rank,
                "min_rank": self.min_rank,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: Union[nn.Linear, Any]) -> CURLinear:
        """Compress a single linear layer into CURLinear."""
        return CURLinear.from_linear(
            linear,
            rank=self.rank,
            method=self.method,
            deterministic=self.deterministic,
            lora_r=self.lora_r,
            lora_alpha=self.lora_alpha,
            freeze_skeletons=self.freeze_skeletons,
        )


@COMPRESSOR_REGISTRY.register("cur")
@COMPRESSOR_REGISTRY.register("cur_factory")
@COMPRESSOR_REGISTRY.register("cur_compressor_factory")
def cur_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR compression."""
    effective_rank = kwargs.pop("rank", rank or getattr(comp_config, "cur_rank", None) or getattr(comp_config, "rank", 64))
    min_rank = kwargs.pop("min_rank", getattr(comp_config, "min_rank", 8))
    method = kwargs.pop("method", getattr(comp_config, "cur_method", "leverage"))
    deterministic = kwargs.pop("deterministic", getattr(comp_config, "deterministic", True))
    max_error = kwargs.pop("max_error", getattr(comp_config, "max_error", 0.20))
    fallback = kwargs.pop("fallback_to_original", getattr(comp_config, "fallback_to_original", True))
    target_names = kwargs.pop("target_names", getattr(comp_config, "cur_target_names", None))
    exclude_names = kwargs.pop("exclude_names", getattr(comp_config, "cur_exclude_names", None))
    return_stats = kwargs.pop("return_stats", kwargs.pop("return_tuple", False))

    compressor = CURCompressor(
        model,
        rank=effective_rank,
        min_rank=min_rank,
        method=method,
        deterministic=deterministic,
        max_error=max_error,
        fallback_to_original=fallback,
        target_names=target_names,
        exclude_names=exclude_names,
        cfg=comp_config,
        **kwargs,
    )
    compressed_model, stats = compressor.compress()
    if return_stats:
        return compressed_model, stats
    return compressed_model
