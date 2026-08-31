"""
CUR-MoE Compressor: apply CUR cross-approximation to Mixture of Experts (MoE) modules and expert layers.
Registered in COMPRESSOR_REGISTRY as 'cur_moe'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

from .cur_linear import (
    CURLinear,
    compute_energy_scores,
    compute_leverage_scores,
    select_indices,
    tikhonov_pinv,
)


@COMPRESSOR_REGISTRY.register("cur_moe")
@COMPRESSOR_REGISTRY.register("cur_moe_compressor")
class CURMoECompressor(BaseCompressor):
    """
    Compresses Mixture of Experts (MoE) layers using CUR cross-approximation.

    Args:
        model: PyTorch model containing MoE modules.
        rank: Target rank for CUR decomposition of expert weights.
        method: 'leverage' (statistical leverage scores) or 'energy' (Frobenius energy).
        cfg: Optional configuration object.
        deterministic: Whether to use deterministic top-k selection (default True).
        **kwargs: Additional parameters for BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        method: str = "leverage",
        cfg: Optional[Any] = None,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur_moe", **kwargs)
        self.rank = rank or getattr(cfg, "cur_moe_rank", None) or getattr(cfg, "rank", 64)
        self.method = method or getattr(cfg, "cur_moe_method", "leverage")
        self.cfg = cfg
        self.deterministic = deterministic
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress expert linear layers in MoE architecture."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        # First scan for standard MoE modules (with module.experts or expert submodules)
        for name, module in list(self.model.named_modules()):
            if hasattr(module, "experts"):
                experts_container = module.experts
                if isinstance(experts_container, (nn.ModuleList, list)):
                    for exp_idx, expert in enumerate(experts_container):
                        for child_name, child_mod in list(expert.named_children()):
                            if isinstance(child_mod, nn.Linear) and not isinstance(child_mod, CURLinear):
                                cur_layer = CURLinear.from_linear(
                                    child_mod,
                                    rank=self.rank,
                                    method=self.method,
                                    deterministic=self.deterministic,
                                )
                                setattr(expert, child_name, cur_layer)
                                self.compressed_layers.append(f"{name}.experts.{exp_idx}.{child_name}")
            elif "expert" in name.lower() and isinstance(module, nn.Linear) and not isinstance(module, CURLinear):
                cur_layer = CURLinear.from_linear(
                    module,
                    rank=self.rank,
                    method=self.method,
                    deterministic=self.deterministic,
                )
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, cur_layer)
                    self.compressed_layers.append(name)

        # If no explicit MoE structure was found, apply to linear layers as fallback.
        # PATCH: Skip attention projection layers — compressing Q/K/V/out simultaneously
        # across all 6 layers causes multiplicative reconstruction error and collapses
        # pretrained representations. Apply CUR only to FFN (feed-forward) layers.
        _ATTN_NAMES = frozenset({
            "query", "key", "value",
            "q_lin", "k_lin", "v_lin",
            "q_proj", "k_proj", "v_proj",
            "out_lin", "out_proj", "dense", "o_proj",
        })
        if len(self.compressed_layers) == 0:
            for name, module in list(self.model.named_modules()):
                last_part = name.split(".")[-1]
                if last_part in _ATTN_NAMES:
                    continue  # preserve pretrained attention weights intact
                if isinstance(module, nn.Linear) and not isinstance(module, CURLinear):
                    max_rank = min(module.in_features, module.out_features)
                    # Use at least rank=128 to retain ≥95% singular energy in FFN layers
                    r = min(max(self.rank, 128), max_rank)
                    cur_layer = CURLinear.from_linear(
                        module,
                        rank=r,
                        method=self.method,
                        deterministic=self.deterministic,
                    )
                    parent = self._get_parent_module(name)
                    if parent is not None:
                        child_name = name.split(".")[-1]
                        setattr(parent, child_name, cur_layer)
                        self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "cur_method": self.method,
                "cur_moe_rank": self.rank,
            }
        )
        return self.model, stats

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("cur_moe")
def cur_moe_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR-MoE compressor."""
    effective_rank = rank or getattr(comp_config, "cur_moe_rank", None) or getattr(comp_config, "rank", 64)
    method = getattr(comp_config, "cur_moe_method", "leverage")
    compressor = CURMoECompressor(model, rank=effective_rank, method=method, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
