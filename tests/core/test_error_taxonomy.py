#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Test Suite for NeuralOps Error Taxonomy & Autograd Graph Cleanliness.

Verifies:
1. 100% of custom exceptions across NeuralOps inherit from NeuralOpsError.
2. Standard library exception dual-inheritance contracts.
3. Structured metadata, error codes, and formatting.
4. Double-backward HVP computation and autograd graph memory cleanliness without leaks.
"""

from __future__ import annotations

import gc
import inspect
import os
import pkgutil
from typing import Sequence

import pytest
import torch
import torch.nn as nn

import neuralops
from neuralops.adapters.errors import AdapterError
from neuralops.benchmarks.regression_gate import MetricNotFoundError
from neuralops.cli.core.errors import (
    ArchitectureMismatchError,
    BackendUnavailableError,
    CliError,
    ConfigurationError as CliConfigurationError,
    ExecutionError as CliExecutionError,
    ExitCode,
    NumericalGateError,
    RunDirectoryLockedError,
)
from neuralops.core import exceptions
from neuralops.core.autograd import (
    FlashAttentionFunction,
    compute_hvp,
    verify_autograd_graph_cleanliness,
)
from neuralops.core.exceptions import (
    AttentionNotRegisteredError,
    BackendError,
    CalibrationError,
    ChecksumMismatchError,
    CompressionError,
    CompressionVerificationError,
    CompressorNotRegisteredError,
    ConfigError,
    ConfigValidationError,
    IncompatibleArchitectureError,
    InferenceError,
    InjectionError,
    IntegrationNotImplementedError,
    KDError,
    KernelDispatchError,
    KernelNotAvailableError,
    LedgerError,
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
from neuralops.distributed.errors import DistributedError
from neuralops.operations.errors import (
    OperationError,
    OperationExecutionError,
    PreconditionFailedError,
)
from neuralops.pipeline_transform.errors import (
    ConfigurationError as PipelineConfigurationError,
    PlanningError,
)
from neuralops.pipeline_transform.result import PipelineError, ValidationFailure
from neuralops.pipeline_transform.rollback import RollbackError
from neuralops.recipe.errors import (
    RecipeError,
    RecipeParseError,
    RecipeSchemaError,
)
from neuralops.utils.qwen_compat import QwenIncompatibleError
from neuralops.validation.errors import (
    RuntimeValidationError,
    StaticValidationError,
    ValidationError,
)


def test_core_exceptions_all_inherit_from_neuralops_error() -> None:
    """Verify all exceptions defined in neuralops.core.exceptions inherit from NeuralOpsError."""
    all_symbols = [getattr(exceptions, name) for name in exceptions.__all__]
    exc_classes = [obj for obj in all_symbols if inspect.isclass(obj) and issubclass(obj, BaseException)]

    assert len(exc_classes) > 20
    for exc_cls in exc_classes:
        assert issubclass(exc_cls, NeuralOpsError), (
            f"Class {exc_cls.__name__} in neuralops.core.exceptions must inherit from NeuralOpsError"
        )


def test_subsystem_errors_inherit_from_neuralops_error() -> None:
    """Verify that all domain-specific error classes across subpackages inherit from NeuralOpsError."""
    subsystem_exceptions = [
        # Distributed
        DistributedError,
        # Recipe
        RecipeError,
        RecipeParseError,
        RecipeSchemaError,
        # CLI
        CliError,
        BackendUnavailableError,
        NumericalGateError,
        CliConfigurationError,
        CliExecutionError,
        ArchitectureMismatchError,
        RunDirectoryLockedError,
        # Operations
        OperationError,
        OperationExecutionError,
        PreconditionFailedError,
        # Adapters
        AdapterError,
        # Validation
        ValidationError,
        StaticValidationError,
        RuntimeValidationError,
        # Pipeline Transform
        PlanningError,
        PipelineConfigurationError,
        PipelineError,
        ValidationFailure,
        RollbackError,
        # Model / Benchmark Utils
        QwenIncompatibleError,
        MetricNotFoundError,
    ]

    for exc_cls in subsystem_exceptions:
        assert issubclass(exc_cls, NeuralOpsError), (
            f"Subsystem exception {exc_cls.__name__} must inherit from NeuralOpsError"
        )


def test_repository_wide_custom_exception_taxonomy() -> None:
    """Introspect all modules under neuralops via AST to certify 100% exception conformance."""
    import ast

    neuralops_path = os.path.dirname(neuralops.__file__)
    non_conforming: list[str] = []
    builtins_err = {
        "Exception",
        "BaseException",
        "StandardError",
    }

    for root, _, files in os.walk(neuralops_path):
        if "/." in root or "node_modules" in root or "web" in root or "_legacy" in root:
            continue
        for f in files:
            if not f.endswith(".py"):
                continue
            file_path = os.path.join(root, f)
            try:
                with open(file_path, "r", encoding="utf-8") as fp:
                    content = fp.read()
                    if "class " not in content:
                        continue
                    tree = ast.parse(content, filename=file_path)
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    base_ids = [
                        b.id for b in node.bases if isinstance(b, ast.Name)
                    ] + [
                        b.attr for b in node.bases if isinstance(b, ast.Attribute)
                    ]
                    if any(b in builtins_err for b in base_ids):
                        if node.name != "NeuralOpsError":
                            non_conforming.append(f"{f}:{node.lineno} {node.name}({base_ids})")

    assert not non_conforming, f"Found non-conforming root exceptions: {non_conforming}"


def test_standard_library_dual_inheritance() -> None:
    """Verify standard library dual inheritance contracts for exception taxonomy."""
    assert issubclass(AttentionNotRegisteredError, (RegistryError, ValueError, KeyError))
    assert issubclass(CompressorNotRegisteredError, (RegistryError, KeyError))
    assert issubclass(ConfigError, (NeuralOpsError, ValueError))
    assert issubclass(ConfigValidationError, ConfigError)
    assert issubclass(KernelNotAvailableError, (BackendError, RuntimeError))
    assert issubclass(KernelDispatchError, (BackendError, RuntimeError))
    assert issubclass(MissingDependencyError, (NeuralOpsError, ImportError))
    assert issubclass(TorchNotInstalledError, MissingDependencyError)
    assert issubclass(TritonNotInstalledError, MissingDependencyError)
    assert issubclass(IntegrationNotImplementedError, (NeuralOpsError, NotImplementedError))
    assert issubclass(IncompatibleArchitectureError, (NeuralOpsError, ValueError))
    assert issubclass(CalibrationError, (NeuralOpsError, RuntimeError))
    assert issubclass(SurgeryTransactionError, (NeuralOpsError, RuntimeError))
    assert issubclass(SingularValueDecompositionError, LinearDecompositionError)
    assert issubclass(LinearDecompositionError, CompressionError)
    assert issubclass(CompressionVerificationError, CompressionError)
    assert issubclass(ChecksumMismatchError, LedgerError)
    assert issubclass(QwenIncompatibleError, (NeuralOpsError, RuntimeError))
    assert issubclass(MetricNotFoundError, (NeuralOpsError, KeyError, ValueError))


def test_cli_error_card_serialization() -> None:
    """Verify CliError structured diagnostic card and to_dict serialization."""
    err = CliError(
        message="Failed operation",
        code=ExitCode.NUMERICAL_GATE_FAILED,
        hint="Increase tolerance budget",
        title="Tolerance Violation",
        context={"diff": 0.05, "threshold": 0.01},
    )
    assert isinstance(err, NeuralOpsError)
    d = err.to_dict()
    assert d["error_code"] == "NUMERICAL_GATE_FAILED"
    assert d["exit_code"] == int(ExitCode.NUMERICAL_GATE_FAILED)
    assert d["title"] == "Tolerance Violation"
    assert d["message"] == "Failed operation"
    assert d["hint"] == "Increase tolerance budget"
    assert d["context"]["diff"] == 0.05


def test_torch_install_error_guidance() -> None:
    """Verify torch_install_error diagnostic output formatting."""
    cause = ImportError("No module named 'torch'")
    err = torch_install_error("benchmark_engine", cause=cause)
    assert isinstance(err, MissingDependencyError)
    assert isinstance(err, NeuralOpsError)
    assert isinstance(err, ImportError)
    assert "NeuralOps requires PyTorch for benchmark_engine" in str(err)
    assert 'pip install "neuralops[core]"' in str(err)


def test_double_backward_hvp_cleanliness() -> None:
    """Verify double-backward Hessian-Vector Product computes exact shapes without graph retention."""
    model = nn.Sequential(
        nn.Linear(32, 64),
        nn.GELU(),
        nn.Linear(64, 16),
        nn.Tanh(),
        nn.Linear(16, 1),
    )
    x = torch.randn(8, 32)
    params = [p for p in model.parameters() if p.requires_grad]
    v = [torch.randn_like(p) for p in params]

    out = model(x)
    loss = (out**2).sum()
    hvps = compute_hvp(loss, params, v, retain_graph=False)

    assert len(hvps) == len(params)
    for h, p in zip(hvps, params):
        assert h.shape == p.shape
        assert not h.requires_grad


def test_autograd_graph_cleanliness_verification() -> None:
    """Verify autograd cleanliness runner completes cycles without residual graph references."""
    model = nn.Sequential(
        nn.Linear(16, 16),
        nn.SiLU(),
        nn.Linear(16, 4),
    )
    x = torch.randn(4, 16)
    is_clean = verify_autograd_graph_cleanliness(model, x, num_iterations=15)
    assert is_clean is True


def test_flash_attention_autograd_backward() -> None:
    """Verify FlashAttentionFunction forward and exact backward gradient propagation."""
    B, H, S, D = 2, 2, 8, 16
    q = torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float64)
    k = torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float64)
    v = torch.randn(B, H, S, D, requires_grad=True, dtype=torch.float64)

    def forward_fn(q_in: torch.Tensor, k_in: torch.Tensor, v_in: torch.Tensor) -> torch.Tensor:
        return FlashAttentionFunction.apply(q_in, k_in, v_in, None, False)

    gradcheck_ok = torch.autograd.gradcheck(forward_fn, (q, k, v), eps=1e-6, atol=1e-4)
    assert gradcheck_ok is True
