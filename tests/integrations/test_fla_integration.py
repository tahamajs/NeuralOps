#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Flash Linear Attention (FLA) framework integration invariants.
"""

from __future__ import annotations

import unittest.mock as mock
import pytest
import torch

from neuralops.core.exceptions import IntegrationNotImplementedError
from neuralops.integrations.fla import (
    FLALayerAdapter,
    fla_is_available,
    list_fla_attentions,
    require_fla,
)


def test_fla_is_available_requires_cuda():
    """fla_is_available must return False if CUDA is unavailable, even when fla package exists."""
    with mock.patch("importlib.util.find_spec", return_value=mock.MagicMock()):
        with mock.patch("torch.cuda.is_available", return_value=False):
            assert not fla_is_available()


def test_fla_is_available_requires_package():
    """fla_is_available must return False if fla package is absent, even when CUDA is True."""
    with mock.patch("importlib.util.find_spec", return_value=None):
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert not fla_is_available()


def test_fla_is_available_true_when_both_present():
    """fla_is_available returns True only when both fla package and CUDA are present."""
    with mock.patch("importlib.util.find_spec", return_value=mock.MagicMock()):
        with mock.patch("torch.cuda.is_available", return_value=True):
            assert fla_is_available()


def test_require_fla_raises_on_cuda_missing():
    """require_fla raises IntegrationNotImplementedError mentioning CUDA when CUDA is missing."""
    with mock.patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(IntegrationNotImplementedError, match="requires an NVIDIA CUDA accelerator"):
            require_fla()


def test_require_fla_raises_on_package_missing():
    """require_fla raises IntegrationNotImplementedError mentioning pip install when package is missing."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("importlib.util.find_spec", return_value=None):
            with pytest.raises(IntegrationNotImplementedError, match="The 'fla' .* package is required"):
                require_fla()


def test_require_fla_succeeds_when_available():
    """require_fla completes silently when both CUDA and fla are available."""
    with mock.patch("torch.cuda.is_available", return_value=True):
        with mock.patch("importlib.util.find_spec", return_value=mock.MagicMock()):
            require_fla()


def test_fla_adapter_clear_error_message_when_missing():
    """FLALayerAdapter instantiation fails closed with an informative error message."""
    with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=False):
        with pytest.raises(IntegrationNotImplementedError, match="pip install 'neuralops\\[fla\\]'"):
            FLALayerAdapter(fla_layer_cls=None, config=None, parent_family="llama")


def test_list_fla_attentions_empty_when_unavailable():
    """list_fla_attentions returns empty list if FLA is unavailable."""
    with mock.patch("neuralops.integrations.fla.fla_is_available", return_value=False):
        assert list_fla_attentions() == []
