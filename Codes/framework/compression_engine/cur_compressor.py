"""
CUR Compressor: replaces Linear layers with CUR decomposition.
Uses leverage score sampling or energy sampling for column/row selection.
Registered in COMPRESSOR_REGISTRY as 'cur'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

from .cur_linear import CURLinear


class CURCompressor(BaseCompressor):
    """
    Compressor that replaces eligible Linear layers with CURLinear.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: int = 64,
        method: str = "leverage",
        cfg: Optional[Any] = None,
        target_names: Optional[List[str]] = None,
        deterministic: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="cur", **kwargs)
        self.rank = rank or getattr(cfg, "cur_rank", None) or getattr(cfg, "rank", 64)
        self.method = method or getattr(cfg, "cur_method", "leverage")
        self.cfg = cfg
        self.target_names = target_names or getattr(cfg, "cur_target_names", None) or getattr(cfg, "target_layers", None)
        self.deterministic = deterministic
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace Linear layers with CURLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if not isinstance(module, nn.Linear) or isinstance(module, CURLinear):
                continue
            # Never compress classifier / lm_head / score
            if any(k in name.lower() for k in ["classifier", "score", "lm_head", "pooler"]):
                continue
            if self.target_names and not any(t in name for t in self.target_names):
                continue

            max_rank = min(module.in_features, module.out_features)
            r = min(self.rank, max_rank)
            if r >= max_rank and max_rank > 4:
                continue

            try:
                cur_linear = CURLinear.from_linear(
                    module, rank=r, method=self.method, deterministic=self.deterministic
                )
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, cur_linear)
                    self.compressed_layers.append(name)
            except Exception as e:
                pass

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "cur_method": self.method,
                "cur_rank": self.rank,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> CURLinear:
        """Compress a single linear layer into CURLinear."""
        return CURLinear.from_linear(
            linear, rank=self.rank, method=self.method, deterministic=self.deterministic
        )

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        """Get parent module by dotted name."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("cur")
def cur_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for CUR compression."""
    effective_rank = rank or getattr(comp_config, "cur_rank", None) or getattr(comp_config, "rank", 64)
    method = getattr(comp_config, "cur_method", "leverage")
    target_names = getattr(comp_config, "cur_target_names", None)
    compressor = CURCompressor(
        model, rank=effective_rank, method=method, target_names=target_names, cfg=comp_config, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
