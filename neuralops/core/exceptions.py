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
    """Base exception for all NeuralOps errors."""


class RegistryError(NeuralOpsError):
    """Raised when a registry lookup or registration fails."""


class AttentionNotRegisteredError(RegistryError, ValueError, KeyError):
    """Raised when a requested attention type is not found in ATTENTION_REGISTRY."""


class CompressorNotRegisteredError(RegistryError, KeyError):
    """Raised when a requested compressor type is not found in COMPRESSOR_REGISTRY."""


class ConfigError(NeuralOpsError, ValueError):
    """Raised when a configuration is invalid or incomplete."""


class BackendError(NeuralOpsError):
    """Raised when no backend can execute the requested operation."""


class KernelNotAvailableError(BackendError, RuntimeError):
    """Raised when a native hardware kernel is requested but not available."""


class KernelDispatchError(BackendError, RuntimeError):
    """Raised when kernel backend dispatch fails across all available levels."""


class MissingDependencyError(NeuralOpsError, ImportError):
    """Raised when an optional dependency required by a feature is not installed."""


class TorchNotInstalledError(MissingDependencyError):
    """Raised when PyTorch is required but not installed."""


class TritonNotInstalledError(MissingDependencyError):
    """Raised when OpenAI Triton is required but not installed."""


class IntegrationNotImplementedError(NeuralOpsError, NotImplementedError):
    """Raised when a framework integration stub is invoked but not implemented."""


class IncompatibleArchitectureError(NeuralOpsError, ValueError):
    """Raised when a model architecture is incompatible with the requested operation."""


class CalibrationError(NeuralOpsError, RuntimeError):
    """Raised when model calibration or parameter reconstruction fails."""


class WeightCopyError(NeuralOpsError):
    """Raised when copying weights from a pretrained attention module fails."""


class NumericalError(NeuralOpsError):
    """Raised when numerical stability is violated (NaN or Inf)."""


class ShapeError(NeuralOpsError):
    """Raised when tensor shapes are incompatible."""


class InjectionError(NeuralOpsError):
    """Raised when injecting a new attention module into a model fails."""


class KDError(NeuralOpsError):
    """Raised when knowledge distillation encounters a fatal error."""


class CompressionError(NeuralOpsError):
    """Raised when model compression fails."""


class InferenceError(NeuralOpsError):
    """Raised when autoregressive inference fails."""


def torch_install_error(feature: str, cause: ImportError | None = None) -> ImportError:
    """Build an actionable error for features that require PyTorch."""
    message = (
        f"NeuralOps requires PyTorch for {feature}, but torch is not installed.\n\n"
        "To enable PyTorch execution, install with the core extra:\n"
        '  pip install "neuralops[core]"\n\n'
        "Or install the extra that matches your hardware acceleration target:\n"
        '  pip install "neuralops[cpu]"    # CPU-only\n'
        '  pip install "neuralops[cuda]"   # NVIDIA GPU (CUDA)\n'
        '  pip install "neuralops[mps]"    # Apple Silicon (MPS)\n'
        '  pip install "neuralops[rocm]"   # AMD GPU (ROCm)\n\n'
        "Alternatively, pre-install your environment's PyTorch build, then run:\n"
        '  pip install "neuralops[core]"'
    )
    if cause is not None:
        message += f"\n\nUnderlying import error: {cause}"
    return MissingDependencyError(message)


# Backward-compatible alias
_torch_install_error = torch_install_error


__all__ = [
    "NeuralOpsError",
    "RegistryError",
    "AttentionNotRegisteredError",
    "CompressorNotRegisteredError",
    "ConfigError",
    "BackendError",
    "KernelNotAvailableError",
    "KernelDispatchError",
    "MissingDependencyError",
    "TorchNotInstalledError",
    "TritonNotInstalledError",
    "IntegrationNotImplementedError",
    "IncompatibleArchitectureError",
    "CalibrationError",
    "WeightCopyError",
    "NumericalError",
    "ShapeError",
    "InjectionError",
    "KDError",
    "CompressionError",
    "InferenceError",
    "torch_install_error",
    "_torch_install_error",
]
