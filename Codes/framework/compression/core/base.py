"""
Base class for all compression methods.
"""

import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class BaseCompressor:
    """
    Abstract base for model compression.

    Subclasses must override ``compress()`` and should call
    ``super().__init__()`` to initialise model tracking.

    Args:
        model: The PyTorch model to compress.
        name: Optional name for the compressor (used in stats).
        **kwargs: Additional compressor‑specific arguments.
    """

    def __init__(self, model: nn.Module, name: Optional[str] = None, **kwargs):
        self.model = model
        self.name = name or self.__class__.__name__
        self.kwargs = kwargs

        # Parameter counts before compression
        self._original_params = self._count_params(model)
        self._compressed_params: Optional[int] = None
        self._compression_stats: Dict[str, Any] = {}

    @staticmethod
    def _count_params(module: nn.Module) -> int:
        """Return the total number of trainable parameters in a module."""
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    def compress(self) -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Apply compression and return the compressed model together with statistics.

        Returns:
            (compressed_model, stats_dict)
        """
        raise NotImplementedError("Subclasses must implement compress()")

    def get_compression_stats(self) -> Dict[str, Any]:
        """
        Return computed compression statistics.
        Call after ``compress()``.
        """
        if self._compressed_params is None:
            self._compressed_params = self._count_params(self.model)
        return self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Build a dictionary with compression metrics."""
        original = self._original_params
        compressed = self._compressed_params or self._count_params(self.model)
        ratio = compressed / original if original > 0 else 1.0
        return {
            "original_params": original,
            "compressed_params": compressed,
            "parameters_saved": original - compressed,
            "compression_ratio": ratio,
            "reduction_percentage": (1.0 - ratio) * 100 if original > 0 else 0.0,
        }

    # ----------------------------------------------------------------
    # Helpers commonly used by child compressors
    # ----------------------------------------------------------------
    @staticmethod
    def _get_parent_module(model: nn.Module, full_name: str) -> Optional[nn.Module]:
        """
        Given a dotted name (e.g. 'encoder.layer.0.attention'), return
        the parent module that holds the last attribute.
        """
        parts = full_name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part, None)
            if parent is None:
                return None
        return parent

    @staticmethod
    def _replace_child(parent: nn.Module, child_name: str, new_module: nn.Module) -> bool:
        """Replace an attribute of *parent* with *new_module*."""
        if hasattr(parent, child_name):
            setattr(parent, child_name, new_module)
            return True
        return False