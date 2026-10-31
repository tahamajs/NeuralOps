#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Offline Fail-Closed Tests for FLA (Flash Linear Attention) Integration Bridge.

Verifies that in an environment where FLA is not installed, all FLA-related
entry points fail closed with IntegrationNotImplementedError and preserve all
canonical invariants.
"""

import sys
import pytest
import torch.nn as nn

from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.integrations.fla import (
    fla_is_available,
    FLALayerAdapter,
    EXTENSION_ATTENTION_REGISTRY,
    SUPPORTED_FLA_ATTENTIONS,
    list_fla_attentions,
)
from neuralops import force_kernel_backend, get_attention_class, replace_attention, list_attentions


def test_fla_is_available_returns_false_when_missing():
    """Verify fla_is_available returns False when fla is not installed."""
    if "fla" not in sys.modules:
        assert not fla_is_available()


@pytest.mark.skipif(
    fla_is_available(),
    reason="FLA is installed; test targets the not-installed fail-closed path",
)
def test_fla_adapter_raises_when_fla_missing():
    """FLALayerAdapter must raise IntegrationNotImplementedError when FLA is missing."""
    with pytest.raises(IntegrationNotImplementedError, match="The 'fla' package is required"):
        FLALayerAdapter(fla_layer_cls=None, config=None, parent_family="llama")


@pytest.mark.skipif(
    fla_is_available(),
    reason="FLA is installed; test targets the not-installed fail-closed path",
)
def test_force_kernel_backend_fla_raises_when_missing():
    """force_kernel_backend('fla') must raise IntegrationNotImplementedError when FLA is missing."""
    with pytest.raises(IntegrationNotImplementedError, match="Cannot force kernel backend 'fla'"):
        with force_kernel_backend("fla"):
            pass


@pytest.mark.skipif(
    fla_is_available(),
    reason="FLA is installed; test targets the not-installed fail-closed path",
)
def test_get_attention_class_fla_prefix_raises_when_missing():
    """get_attention_class('fla:gla') must raise IntegrationNotImplementedError when FLA is missing."""
    with pytest.raises(IntegrationNotImplementedError, match="Cannot load FLA attention 'fla:gla'"):
        get_attention_class("fla:gla")


@pytest.mark.skipif(
    fla_is_available(),
    reason="FLA is installed; test targets the not-installed fail-closed path",
)
def test_replace_attention_fla_prefix_raises_when_missing():
    """replace_attention with fla: prefix must raise IntegrationNotImplementedError when FLA is missing."""
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 16)

    model = DummyModel()
    with pytest.raises(IntegrationNotImplementedError, match="Cannot replace attention with 'fla:gla'"):
        replace_attention(model, attention="fla:gla")


def test_list_fla_attentions_empty_when_missing():
    """list_fla_attentions() returns [] when FLA is missing."""
    if not fla_is_available():
        assert list_fla_attentions() == []


def test_attention_registry_invariant_unaffected():
    """Preserve canonical ATTENTION_REGISTRY invariant and extension isolation."""
    # Ensure extension entries are strictly kept in EXTENSION_ATTENTION_REGISTRY
    assert len(EXTENSION_ATTENTION_REGISTRY) >= len(SUPPORTED_FLA_ATTENTIONS)
    for k in EXTENSION_ATTENTION_REGISTRY:
        assert k.startswith("fla:")
        assert k not in ATTENTION_REGISTRY


def test_lazy_import_no_top_level_fla_leak():
    """Verify importing neuralops or neuralops.integrations.fla does not import fla into sys.modules."""
    if not fla_is_available():
        assert "fla" not in sys.modules
        assert "fla.ops" not in sys.modules
        assert "fla.layers" not in sys.modules


def test_list_attentions_extension_flag():
    """list_attentions(include_extensions=False) returns canonical entries without extensions."""
    canonical = list_attentions(include_extensions=False)
    for name in canonical:
        assert not name.startswith("fla:")
