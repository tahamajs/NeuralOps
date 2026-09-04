"""
Tensor Train Compressor: replaces all nn.Linear layers with TTLinearLayer.
Registered in COMPRESSOR_REGISTRY as "tt".
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY
from framework.utils.compat import should_compress_linear

from .tt_linear import TTLinearLayer


class TTCompressor(BaseCompressor):
    """
    Compressor that replaces nn.Linear layers with Tensor-Train (TTLinearLayer) modules.

    Args:
        model: PyTorch model to compress.
        rank: Rank parameter (alias for tt_rank).
        tt_rank: TT-rank bound for tensor-train cores.
        cfg: Optional configuration object.
        target_layers: Optional list of layer names or substrings to compress.
        **kwargs: Additional arguments passed to BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        tt_rank: int = 16,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="tt", **kwargs)
        self.tt_rank = rank or getattr(cfg, "tt_rank", None) or getattr(cfg, "rank", None) or tt_rank
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[Dict[str, Any]] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Replace linear layers with TTLinearLayer and return (model, stats)."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, TTLinearLayer):
                if not should_compress_linear(self.model, name):
                    continue
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                tt_layer = TTLinearLayer.from_linear(module, tt_rank=self.tt_rank)
                parent = self._get_parent_module(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, tt_layer)
                    self.compressed_layers.append(
                        {
                            "name": name,
                            "original_shape": (module.in_features, module.out_features),
                            "tt_rank": self.tt_rank,
                        }
                    )

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layers": len(self.compressed_layers),
                "tt_rank": self.tt_rank,
            }
        )
        return self.model, stats

    def compress_linear(self, linear_layer: nn.Linear) -> TTLinearLayer:
        """Compress a single linear layer into TTLinearLayer."""
        return TTLinearLayer.from_linear(linear_layer, tt_rank=self.tt_rank)

    def _get_parent_module(self, full_name: str) -> Optional[nn.Module]:
        """Retrieve parent module by dotted path."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("tt")
def tt_compressor_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for Tensor Train compressor."""
    tt_rank = rank or getattr(comp_config, "tt_rank", None) or getattr(comp_config, "rank", 16)
    compressor = TTCompressor(model, tt_rank=tt_rank, cfg=comp_config, **kwargs)
    compressed_model, _ = compressor.compress()
    return compressed_model
