"""
NeuralOps Exception Hierarchy

Structured exception hierarchy providing actionable diagnostic messages across all core modules.

Author: Taha Majlesi
"""

from __future__ import annotations
from typing import Optional, Any


class NeuralOpsError(Exception):
    """Base exception for all neuralops errors."""
    pass


class ConfigError(NeuralOpsError):
    """Raised when a configuration is invalid or incomplete."""
    pass


class WeightCopyError(NeuralOpsError):
    """Raised when copying weights from a pretrained attention module fails."""
    pass


class BackendError(NeuralOpsError):
    """Raised when no backend can execute the requested operation."""
    pass


class NumericalError(NeuralOpsError):
    """Raised when numerical stability is violated (NaN or Inf)."""
    pass


class ShapeError(NeuralOpsError):
    """Raised when tensor shapes are incompatible."""
    pass


class RegistryError(NeuralOpsError):
    """Raised when a registry lookup or registration fails."""
    pass


class InjectionError(NeuralOpsError):
    """Raised when injecting a new attention module into a model fails."""
    pass


class KDError(NeuralOpsError):
    """Raised when knowledge distillation encounters a fatal error."""
    pass


class CompressionError(NeuralOpsError):
    """Raised when model compression fails."""
    pass


class InferenceError(NeuralOpsError):
    """Raised when autoregressive inference fails."""
    pass


__all__ = [
    "NeuralOpsError",
    "ConfigError",
    "WeightCopyError",
    "BackendError",
    "NumericalError",
    "ShapeError",
    "RegistryError",
    "InjectionError",
    "KDError",
    "CompressionError",
    "InferenceError",
]
