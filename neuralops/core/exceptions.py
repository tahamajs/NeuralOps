#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Exception Hierarchy

Structured exception hierarchy providing actionable diagnostic messages across all core modules.
"""

from __future__ import annotations
from typing import Optional, Any


class NeuralOpsError(Exception):
    """Base exception for all neuralops errors."""


class ConfigError(NeuralOpsError):
    """Raised when a configuration is invalid or incomplete."""


class WeightCopyError(NeuralOpsError):
    """Raised when copying weights from a pretrained attention module fails."""


class BackendError(NeuralOpsError):
    """Raised when no backend can execute the requested operation."""


class NumericalError(NeuralOpsError):
    """Raised when numerical stability is violated (NaN or Inf)."""


class ShapeError(NeuralOpsError):
    """Raised when tensor shapes are incompatible."""


class RegistryError(NeuralOpsError):
    """Raised when a registry lookup or registration fails."""


class InjectionError(NeuralOpsError):
    """Raised when injecting a new attention module into a model fails."""


class KDError(NeuralOpsError):
    """Raised when knowledge distillation encounters a fatal error."""


class CompressionError(NeuralOpsError):
    """Raised when model compression fails."""


class InferenceError(NeuralOpsError):
    """Raised when autoregressive inference fails."""


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
