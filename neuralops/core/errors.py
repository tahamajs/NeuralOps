#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Exception hierarchy and dependency installation error builders for NeuralOps.
"""

from __future__ import annotations


class NeuralOpsError(Exception):
    """Base class for all NeuralOps exceptions."""


class MissingDependencyError(ImportError, NeuralOpsError):
    """Raised when an optional dependency required by a feature is not installed."""


class TorchNotInstalledError(MissingDependencyError):
    """Raised when PyTorch is required but not installed."""


class TritonNotInstalledError(MissingDependencyError):
    """Raised when OpenAI Triton is required but not installed."""


class AttentionNotRegisteredError(KeyError, NeuralOpsError):
    """Raised when an requested attention type is not found in ATTENTION_REGISTRY."""


class CompressorNotRegisteredError(KeyError, NeuralOpsError):
    """Raised when a requested compressor type is not found in COMPRESSOR_REGISTRY."""


class KernelNotAvailableError(RuntimeError, NeuralOpsError):
    """Raised when a native hardware kernel is requested but not available."""


class IntegrationNotImplementedError(NotImplementedError, NeuralOpsError):
    """Raised when a framework integration stub is invoked but not implemented."""


class IncompatibleArchitectureError(ValueError, NeuralOpsError):
    """Raised when a model architecture is incompatible with the requested operation."""


class KernelDispatchError(RuntimeError, NeuralOpsError):
    """Raised when kernel backend dispatch fails across all available levels."""


class CalibrationError(RuntimeError, NeuralOpsError):
    """Raised when model calibration or parameter reconstruction fails."""


def torch_install_error(feature: str, cause: ImportError | None = None) -> ImportError:
    """Build an actionable error for features that require PyTorch."""
    message = (
        f"NeuralOps requires PyTorch for {feature}, but torch is not installed.\n\n"
        "Install the extra that matches your hardware:\n"
        '  pip install "neuralops[cpu]"    # CPU-only\n'
        '  pip install "neuralops[cuda]"   # NVIDIA GPU\n'
        '  pip install "neuralops[mps]"    # Apple Silicon\n'
        '  pip install "neuralops[rocm]"   # AMD GPU\n\n'
        "Alternatively, install a PyTorch build selected for your hardware, then install "
        '"neuralops[core]".'
    )
    if cause is not None:
        message += f"\n\nUnderlying import error: {cause}"
    return MissingDependencyError(message)


# Backward-compatible alias
_torch_install_error = torch_install_error
