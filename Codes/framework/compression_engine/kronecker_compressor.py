"""
Kronecker Product Compressor: replaces nn.Linear with KroneckerLinear.
Registered in COMPRESSOR_REGISTRY as 'kronecker'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

from .kronecker_linear import KroneckerLinear


class KroneckerCompressor(BaseCompressor):
    """
    Compresses a model by replacing every eligible nn.Linear layer with a
    Kronecker product decomposition (A ⊗ B), falling back to low-rank SVD
    when dimensions do not allow composite factorization.

    Args:
        model: PyTorch model to compress (optional if instantiated as standalone engine).
        rank: Rank parameter used for fallback or rank constraints.
        cfg: Optional configuration object.
        target_layers: Optional list of layer names or substrings to compress.
        **kwargs: Additional parameters passed to BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="kronecker", **kwargs)
        self.rank = rank or getattr(cfg, "kronecker_rank", None) or getattr(cfg, "rank", None)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Compress the model by substituting Linear layers with KroneckerLinear.

        Returns:
            Tuple of (compressed_model, stats_dictionary).
        """
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, KroneckerLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                new_layer = KroneckerLinear.from_linear(module, rank=self.rank)
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, new_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats["compressed_layer_count"] = len(self.compressed_layers)
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> KroneckerLinear:
        """Compress a single linear layer into KroneckerLinear."""
        return KroneckerLinear.from_linear(linear, rank=self.rank)

    def _get_parent(self, full_name: str) -> Optional[nn.Module]:
        """Traverse model hierarchy to retrieve the parent module."""
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("kronecker")
def kronecker_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Kronecker compressor factory function."""
    compressor = KroneckerCompressor(model, rank=rank, cfg=comp_config, **kwargs)
    compressed_model, stats = compressor.compress()
    return compressed_model
