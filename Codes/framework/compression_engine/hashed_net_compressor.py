"""
HashedNet Compressor: replaces nn.Linear with HashedLinear.
Registered in COMPRESSOR_REGISTRY as 'hashed_net'.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from framework.core.base import BaseCompressor
from framework.core.registry import COMPRESSOR_REGISTRY

from .hashed_linear import HashedLinear


class HashedNetCompressor(BaseCompressor):
    """
    Compress a model by replacing every nn.Linear layer with HashedLinear,
    which shares weights globally through compact hash tables.

    Args:
        model: PyTorch model.
        rank: Rank parameter (optionally maps to bucket budget).
        n_buckets: Number of hash buckets (size of shared weight table).
        use_sign: Whether to use auxiliary sign hashing (unbiased trick).
        cfg: Optional configuration object.
        target_layers: Optional list of layer names or substrings to compress.
        **kwargs: Passed to BaseCompressor.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        rank: Optional[int] = None,
        n_buckets: int = 1024,
        use_sign: bool = True,
        cfg: Optional[Any] = None,
        target_layers: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model if model is not None else nn.Identity(), name="hashed_net", **kwargs)
        if rank is not None:
            self.n_buckets = max(16, rank * 32)
        else:
            self.n_buckets = (
                getattr(cfg, "hashed_net_buckets", None)
                or getattr(cfg, "n_buckets", None)
                or n_buckets
            )
        self.rank = rank or getattr(cfg, "rank", None)
        self.use_sign = use_sign if cfg is None else getattr(cfg, "use_sign", use_sign)
        self.cfg = cfg
        self.target_layers = target_layers or getattr(cfg, "target_layers", None)
        self.compressed_layers: List[str] = []

    def compress(self, model: Optional[nn.Module] = None) -> Tuple[nn.Module, Dict[str, Any]]:
        """Compress the target model by substituting Linear layers with HashedLinear."""
        if model is not None:
            self.model = model
            self._original_params = self._count_params(self.model)

        self.compressed_layers = []
        for name, module in list(self.model.named_modules()):
            if isinstance(module, nn.Linear) and not isinstance(module, HashedLinear):
                if self.target_layers and not any(t in name for t in self.target_layers):
                    continue

                new_layer = HashedLinear.from_linear(
                    module, n_buckets=self.n_buckets, use_sign=self.use_sign
                )
                parent = self._get_parent(name)
                if parent is not None:
                    child_name = name.split(".")[-1]
                    setattr(parent, child_name, new_layer)
                    self.compressed_layers.append(name)

        self._compressed_params = self._count_params(self.model)
        stats = self.get_compression_stats()
        stats.update(
            {
                "compressed_layer_count": len(self.compressed_layers),
                "n_buckets": self.n_buckets,
                "use_sign": self.use_sign,
            }
        )
        return self.model, stats

    def compress_linear(self, linear: nn.Linear) -> HashedLinear:
        """Compress a single linear layer into HashedLinear."""
        return HashedLinear.from_linear(
            linear, n_buckets=self.n_buckets, use_sign=self.use_sign
        )

    def _get_parent(self, full_name: str) -> Optional[nn.Module]:
        parts = full_name.split(".")
        parent = self.model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent


@COMPRESSOR_REGISTRY.register("hashed_net")
def hashed_net_factory(
    model: nn.Module,
    comp_config: Optional[Any] = None,
    rank: Optional[int] = None,
    **kwargs: Any,
) -> Union[nn.Module, Tuple[nn.Module, Dict[str, Any]]]:
    """Factory function for HashedNet compressor."""
    n_buckets = getattr(comp_config, "hashed_net_buckets", None) or getattr(comp_config, "n_buckets", 1024)
    use_sign = getattr(comp_config, "use_sign", True)
    compressor = HashedNetCompressor(
        model, rank=rank, n_buckets=n_buckets, use_sign=use_sign, cfg=comp_config, **kwargs
    )
    compressed_model, _ = compressor.compress()
    return compressed_model
