#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Contract tests for NeuralOps Exception Taxonomy & Error Hierarchy Consistency.

Validates that all domain-specific exceptions across neuralops inherit from
NeuralOpsError, observe standard exception ergonomics, and support error codes.
"""

from __future__ import annotations

import inspect
import pytest

from neuralops.core.exceptions import NeuralOpsError
import neuralops.core.exceptions as core_exc
import neuralops.surgery.director as surgery_mod
import neuralops.benchmarks.results_v2.record as record_mod
import neuralops.recipe.errors as recipe_errors
import neuralops.cli.core.errors as cli_errors
import neuralops.operations.errors as operations_errors
import neuralops.adapters.errors as adapters_errors
import neuralops.validation.errors as validation_errors
import neuralops.pipeline_transform.errors as pipeline_errors
import neuralops.distributed.errors as distributed_errors
import neuralops.benchmarks.config_v2.loader as config_v2_loader
import neuralops.hub.uploader as hub_uploader


def test_core_exceptions_inherit_from_neuralops_error() -> None:
    """Verify all custom exceptions in core inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(core_exc, inspect.isclass):
        if issubclass(obj, Exception) and obj not in {
            Exception,
            ValueError,
            TypeError,
            KeyError,
            NeuralOpsError,
        }:
            assert issubclass(
                obj, NeuralOpsError
            ), f"Class {name} does not inherit from NeuralOpsError"


def test_surgery_exceptions_inherit_from_neuralops_error() -> None:
    """Verify surgery module exceptions inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(surgery_mod, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name:
            assert issubclass(
                obj, (NeuralOpsError, ValueError, TypeError)
            ), f"Surgery exception {name} not in hierarchy"


def test_exception_has_standard_attributes() -> None:
    """Verify NeuralOpsError supports message and error_code."""
    err = NeuralOpsError("Operation failed", error_code="OP_FAILED")
    assert str(err) == "Operation failed"
    assert getattr(err, "error_code", None) == "OP_FAILED"
    assert getattr(err, "message", None) == "Operation failed"

    # Default construction
    default_err = NeuralOpsError()
    assert str(default_err) == ""
    assert default_err.error_code is None


def test_recipe_exceptions_inherit_from_neuralops_error() -> None:
    """Verify recipe error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(recipe_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Recipe exception {name} does not inherit from NeuralOpsError"


def test_cli_exceptions_inherit_from_neuralops_error() -> None:
    """Verify CLI error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(cli_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"CLI exception {name} does not inherit from NeuralOpsError"


def test_operations_exceptions_inherit_from_neuralops_error() -> None:
    """Verify operations error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(operations_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Operations exception {name} does not inherit from NeuralOpsError"


def test_adapters_exceptions_inherit_from_neuralops_error() -> None:
    """Verify adapter error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(adapters_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Adapter exception {name} does not inherit from NeuralOpsError"


def test_validation_exceptions_inherit_from_neuralops_error() -> None:
    """Verify validation error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(validation_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Validation exception {name} does not inherit from NeuralOpsError"


def test_pipeline_exceptions_inherit_from_neuralops_error() -> None:
    """Verify pipeline transform error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(pipeline_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Pipeline exception {name} does not inherit from NeuralOpsError"


def test_distributed_exceptions_inherit_from_neuralops_error() -> None:
    """Verify distributed error hierarchy inherits from NeuralOpsError."""
    for name, obj in inspect.getmembers(distributed_errors, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Distributed exception {name} does not inherit from NeuralOpsError"


def test_results_v2_exceptions_inherit_from_neuralops_error() -> None:
    """Verify results_v2 record exceptions inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(record_mod, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Results_v2 exception {name} does not inherit from NeuralOpsError"


def test_config_v2_exceptions_inherit_from_neuralops_error() -> None:
    """Verify config_v2 loader exceptions inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(config_v2_loader, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Config_v2 exception {name} does not inherit from NeuralOpsError"


def test_hub_exceptions_inherit_from_neuralops_error() -> None:
    """Verify hub uploader exceptions inherit from NeuralOpsError."""
    for name, obj in inspect.getmembers(hub_uploader, inspect.isclass):
        if issubclass(obj, Exception) and "Error" in name and getattr(obj, "__module__", "").startswith("neuralops"):
            assert issubclass(
                obj, NeuralOpsError
            ), f"Hub exception {name} does not inherit from NeuralOpsError"


def test_neuralops_exceptions_shim() -> None:
    """Verify neuralops.exceptions re-exports NeuralOpsError and core exceptions."""
    import neuralops.exceptions as exc_mod

    assert hasattr(exc_mod, "NeuralOpsError")
    assert issubclass(exc_mod.NeuralOpsError, Exception)
    assert exc_mod.NeuralOpsError is NeuralOpsError


def test_all_custom_exceptions_catchable_via_neuralops_error() -> None:
    """Verify raising any custom domain exception is catchable via except NeuralOpsError."""
    exception_classes = [
        core_exc.RegistryError,
        core_exc.AttentionNotRegisteredError,
        core_exc.CompressorNotRegisteredError,
        core_exc.ConfigError,
        core_exc.ConfigurationError,
        core_exc.BackendError,
        core_exc.BackendUnavailableError,
        core_exc.KernelNotAvailableError,
        core_exc.KernelDispatchError,
        core_exc.MissingDependencyError,
        core_exc.TorchNotInstalledError,
        core_exc.TritonNotInstalledError,
        core_exc.IntegrationNotImplementedError,
        core_exc.IncompatibleArchitectureError,
        core_exc.CalibrationError,
        core_exc.WeightCopyError,
        core_exc.NumericalError,
        core_exc.ShapeError,
        core_exc.InjectionError,
        core_exc.SurgeryTransactionError,
        core_exc.InvariantViolationError,
        core_exc.KDError,
        core_exc.CompressionError,
        core_exc.CompressionVerificationError,
        core_exc.LinearDecompositionError,
        core_exc.SingularValueDecompositionError,
        core_exc.InferenceError,
        core_exc.LedgerError,
        core_exc.ChecksumMismatchError,
        core_exc.ConfigValidationError,
        core_exc.BenchmarkError,
        core_exc.BenchmarkExecutionError,
        core_exc.WorkerCrashError,
        core_exc.HubUploadError,
        core_exc.ChecksumVerificationError,
        core_exc.RecipeSyntaxError,
        recipe_errors.RecipeError,
        recipe_errors.RecipeParseError,
        recipe_errors.RecipeSchemaError,
        cli_errors.CliError,
        cli_errors.BackendUnavailableError,
        cli_errors.NumericalGateError,
        cli_errors.ConfigurationError,
        cli_errors.ExecutionError,
        cli_errors.ArchitectureMismatchError,
        cli_errors.RunDirectoryLockedError,
        operations_errors.OperationError,
        operations_errors.OperationExecutionError,
        operations_errors.PreconditionFailedError,
        adapters_errors.AdapterError,
        validation_errors.ValidationError,
        validation_errors.StaticValidationError,
        validation_errors.RuntimeValidationError,
        pipeline_errors.PlanningError,
        pipeline_errors.ConfigurationError,
        distributed_errors.DistributedError,
    ]

    for exc_cls in exception_classes:
        caught = False
        try:
            # Instantiate with flexible signature
            try:
                raise exc_cls("Test error message")
            except TypeError:
                try:
                    raise exc_cls("Test error message", "E_TEST_CODE")
                except TypeError:
                    raise exc_cls()
        except NeuralOpsError:
            caught = True
        except Exception:
            caught = False

        assert caught, f"Exception class {exc_cls.__name__} was not caught by except NeuralOpsError!"
