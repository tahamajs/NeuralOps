"""
Base class for compression methods.
Every compressor must inherit from BaseCompressor and implement `compress()`.
"""

import torch.nn as nn
from typing import Dict, Any, Tuple, Optional


class BaseCompressor:
    """
    Abstract base for model compression.

    Subclasses must override `compress()` and should call `super().__init__()`
    to initialize model tracking.

    Args:
        model: The PyTorch model to compress.
        name: Optional name for the compressor (used in stats).
        **kwargs: Additional compressor-specific arguments.
    """

    def __init__(self, model: nn.Module, name: Optional[str] = None, **kwargs):
        self.model = model
        self.name = name or self.__class__.__name__
        self.kwargs = kwargs

        # Parameter counts before compression
        self._original_params = self._count_params(model)
        self._compressed_params = None
        self._compression_stats = {}

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
        Call after `compress()`.
        """
        if self._compressed_params is None:
            # If compress hasn't been called, compute on the fly
            self._compressed_params = self._count_params(self.model)
        return self._compute_stats()

    def _compute_stats(self) -> Dict[str, Any]:
        """Build a dictionary with compression metrics."""
        original = self._original_params
        compressed = self._compressed_params or self._count_params(self.model)
        return {
            'original_params': original,
            'compressed_params': compressed,
            'parameters_saved': original - compressed,
            'compression_ratio': compressed / original if original > 0 else 1.0,
            'reduction_percentage': (1 - compressed / original) * 100 if original > 0 else 0,
        }