#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for single exception hierarchy consolidation across neuralops.core.exceptions
and neuralops.core.errors (audit finding F-81).
"""

from __future__ import annotations

import pytest

import neuralops
import neuralops.api as napi
import neuralops.core.errors as n_errors
import neuralops.core.exceptions as n_exceptions
from neuralops.core.exceptions import (
    AttentionNotRegisteredError,
    CompressorNotRegisteredError,
    NeuralOpsError,
    RegistryError,
)
from neuralops.core.registry import (
    create_attention,
    create_compressor,
    get_attention,
    get_compressor,
)


def test_both_modules_share_one_neural_ops_error():
    """Verify that neuralops.core.exceptions and neuralops.core.errors share the exact same classes."""
    assert n_exceptions.NeuralOpsError is n_errors.NeuralOpsError
    assert n_exceptions.RegistryError is n_errors.RegistryError
    assert n_exceptions.AttentionNotRegisteredError is n_errors.AttentionNotRegisteredError
    assert n_exceptions.CompressorNotRegisteredError is n_errors.CompressorNotRegisteredError
    assert n_exceptions.MissingDependencyError is n_errors.MissingDependencyError
    assert n_exceptions.TorchNotInstalledError is n_errors.TorchNotInstalledError
    assert n_exceptions.TritonNotInstalledError is n_errors.TritonNotInstalledError
    assert n_exceptions.KernelNotAvailableError is n_errors.KernelNotAvailableError
    assert n_exceptions.IntegrationNotImplementedError is n_errors.IntegrationNotImplementedError
    assert n_exceptions.IncompatibleArchitectureError is n_errors.IncompatibleArchitectureError
    assert n_exceptions.KernelDispatchError is n_errors.KernelDispatchError
    assert n_exceptions.CalibrationError is n_errors.CalibrationError
    assert n_exceptions.torch_install_error is n_errors.torch_install_error


def test_except_neural_ops_error_catches_both_families():
    """Verify that catching NeuralOpsError catches errors from both legacy and new names."""
    try:
        raise n_errors.AttentionNotRegisteredError("Missing attention")
    except NeuralOpsError as e:
        assert "Missing attention" in str(e)

    try:
        raise n_exceptions.ConfigError("Invalid config")
    except n_errors.NeuralOpsError as e:
        assert "Invalid config" in str(e)


def test_unknown_attention_raises_attention_not_registered_error():
    """Verify that unknown attention names raise AttentionNotRegisteredError from all 3 public paths."""
    # 1. neuralops.api.replace_attention
    with pytest.raises(AttentionNotRegisteredError):
        napi.replace_attention(None, attention="unknown_attention_xyz_123")

    # 2. neuralops.create_attention (top-level re-export)
    with pytest.raises(AttentionNotRegisteredError):
        neuralops.create_attention("unknown_attention_xyz_123", dim=64)

    # 3. neuralops.core.registry.create_attention
    with pytest.raises(AttentionNotRegisteredError):
        create_attention("unknown_attention_xyz_123", dim=64)

    # Also test get_attention
    with pytest.raises(AttentionNotRegisteredError):
        get_attention("unknown_attention_xyz_123")


def test_unknown_compressor_raises_compressor_not_registered_error():
    """Verify that unknown compressor names raise CompressorNotRegisteredError from public paths."""
    # 1. neuralops.create_compressor
    with pytest.raises(CompressorNotRegisteredError):
        neuralops.create_compressor("unknown_compressor_xyz_123")

    # 2. neuralops.core.registry.create_compressor
    with pytest.raises(CompressorNotRegisteredError):
        create_compressor("unknown_compressor_xyz_123")

    # Also test get_compressor
    with pytest.raises(CompressorNotRegisteredError):
        get_compressor("unknown_compressor_xyz_123")


def test_unknown_attention_still_caught_as_value_error():
    """Verify backward compatibility: AttentionNotRegisteredError inherits from ValueError and KeyError."""
    assert issubclass(AttentionNotRegisteredError, ValueError)
    assert issubclass(AttentionNotRegisteredError, RegistryError)
    assert issubclass(AttentionNotRegisteredError, NeuralOpsError)

    with pytest.raises(ValueError):
        neuralops.create_attention("unknown_attention_xyz_123", dim=64)

    with pytest.raises(NeuralOpsError):
        neuralops.create_attention("unknown_attention_xyz_123", dim=64)


def test_unknown_compressor_still_caught_as_key_error():
    """Verify backward compatibility: CompressorNotRegisteredError inherits from KeyError."""
    assert issubclass(CompressorNotRegisteredError, KeyError)
    assert issubclass(CompressorNotRegisteredError, RegistryError)
    assert issubclass(CompressorNotRegisteredError, NeuralOpsError)

    with pytest.raises(KeyError):
        neuralops.create_compressor("unknown_compressor_xyz_123")

    with pytest.raises(NeuralOpsError):
        neuralops.create_compressor("unknown_compressor_xyz_123")
