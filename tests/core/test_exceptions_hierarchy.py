#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.exceptions hierarchy and diagnostic messages."""

from __future__ import annotations

import inspect
import pytest

from neuralops.core import exceptions
from neuralops.core.exceptions import (
    AttentionNotRegisteredError,
    BackendError,
    CalibrationError,
    CompressionError,
    CompressionVerificationError,
    CompressorNotRegisteredError,
    ConfigError,
    IncompatibleArchitectureError,
    InferenceError,
    InjectionError,
    IntegrationNotImplementedError,
    KDError,
    KernelDispatchError,
    KernelNotAvailableError,
    LinearDecompositionError,
    MissingDependencyError,
    NeuralOpsError,
    NumericalError,
    RegistryError,
    ShapeError,
    SingularValueDecompositionError,
    SurgeryTransactionError,
    TorchNotInstalledError,
    TritonNotInstalledError,
    WeightCopyError,
    torch_install_error,
)


def test_exception_inheritance_base() -> None:
    """Verify all custom exceptions inherit from NeuralOpsError."""
    all_symbols = [getattr(exceptions, name) for name in exceptions.__all__]
    exc_classes = [obj for obj in all_symbols if inspect.isclass(obj) and issubclass(obj, BaseException)]

    for exc_cls in exc_classes:
        assert issubclass(exc_cls, NeuralOpsError), f"{exc_cls} must inherit from NeuralOpsError"


def test_standard_library_exception_compatibility() -> None:
    """Verify NeuralOps exceptions maintain standard library dual inheritance."""
    assert issubclass(AttentionNotRegisteredError, (RegistryError, ValueError, KeyError))
    assert issubclass(CompressorNotRegisteredError, (RegistryError, KeyError))
    assert issubclass(ConfigError, (NeuralOpsError, ValueError))
    assert issubclass(KernelNotAvailableError, (BackendError, RuntimeError))
    assert issubclass(KernelDispatchError, (BackendError, RuntimeError))
    assert issubclass(MissingDependencyError, (NeuralOpsError, ImportError))
    assert issubclass(TorchNotInstalledError, MissingDependencyError)
    assert issubclass(TritonNotInstalledError, MissingDependencyError)
    assert issubclass(IntegrationNotImplementedError, (NeuralOpsError, NotImplementedError))
    assert issubclass(IncompatibleArchitectureError, (NeuralOpsError, ValueError))
    assert issubclass(SurgeryTransactionError, (NeuralOpsError, RuntimeError))
    assert issubclass(SingularValueDecompositionError, LinearDecompositionError)
    assert issubclass(LinearDecompositionError, CompressionError)


def test_torch_install_error_formatting() -> None:
    """Verify torch_install_error produces clear actionable instruction."""
    cause = ImportError("No module named 'torch'")
    err = torch_install_error("benchmark_suite", cause=cause)
    assert isinstance(err, MissingDependencyError)
    assert "NeuralOps requires PyTorch for benchmark_suite" in str(err)
    assert 'pip install "neuralops[core]"' in str(err)
    assert "Underlying import error" in str(err)


def test_all_exports_present() -> None:
    """Verify __all__ is complete and free of dangling symbols."""
    for sym in exceptions.__all__:
        assert hasattr(exceptions, sym), f"Symbol '{sym}' listed in __all__ but missing from module"
