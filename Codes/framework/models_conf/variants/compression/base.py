"""
Compression base – abstract interface for model compression strategies.

Defines the contract for compression implementations (distillation, pruning, quantization, etc.).
"""

from abc import ABC, abstractmethod

from framework.core.registry import Registry

COMPRESSOR_REGISTRY = Registry("compressors")


class BaseCompressor(ABC):
    @abstractmethod
    def apply(self, model, **kwargs) -> None:
        """Return the compressed model."""
        pass
