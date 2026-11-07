#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying single-inspection factory signatures and proper error propagation (F-85)."""

from __future__ import annotations

import pytest
import torch.nn as nn

from neuralops.core.registry import (
    ATTENTION_REGISTRY,
    COMPRESSOR_REGISTRY,
    create_attention,
    create_compressor,
)


class DummyCustomErrorAttention(nn.Module):
    """Test attention module whose __init__ deliberately raises a distinctive TypeError."""

    def __init__(self, embed_dim: int, num_heads: int, **kwargs: object) -> None:
        super().__init__()
        raise TypeError("DISTINCTIVE_INNER_TYPE_ERROR_MESSAGE")


class StrictTestAttention(nn.Module):
    """Test attention module with strict parameters (no **kwargs)."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads


def test_inner_type_error_not_misread_as_signature() -> None:
    """Assert that a TypeError raised inside __init__ propagates unchanged to caller."""
    test_key = "__test_custom_inner_error_attention__"
    ATTENTION_REGISTRY.register(test_key)(DummyCustomErrorAttention)
    try:
        with pytest.raises(TypeError) as exc_info:
            create_attention(test_key, dim=64, num_heads=4)
        assert "DISTINCTIVE_INNER_TYPE_ERROR_MESSAGE" in str(exc_info.value)
    finally:
        ATTENTION_REGISTRY._dict.pop(test_key, None)


def test_create_attention_wrong_args_clear_error() -> None:
    """Assert that mismatched arguments produce a clear TypeError naming class and signature."""
    test_key = "__test_strict_attention__"
    ATTENTION_REGISTRY.register(test_key)(StrictTestAttention)
    try:
        with pytest.raises(TypeError) as exc_info:
            # Pass invalid unexpected argument that fails signature binding
            create_attention(test_key, dim=64, num_heads=4, unexpected_extra_arg="invalid")
        err_str = str(exc_info.value)
        assert "StrictTestAttention" in err_str
        assert "signature" in err_str.lower()
    finally:
        ATTENTION_REGISTRY._dict.pop(test_key, None)


def test_create_compressor_no_dummy_host() -> None:
    """Verify that create_compressor with model=None never builds a dummy host."""
    # Test across registered compressors: when model=None, it either raises TypeError (if required)
    # or returns an unattached compressor whose model is NOT nn.Sequential(nn.Linear(64,64))
    for name in list(COMPRESSOR_REGISTRY.keys())[:5]:
        try:
            comp = create_compressor(name, model=None, rank=16)
            host_model = getattr(comp, "model", None)
            if host_model is not None:
                is_dummy = isinstance(host_model, nn.Sequential) and any(
                    isinstance(m, nn.Linear) and m.in_features == 64 for m in host_model
                )
                assert not is_dummy, f"Compressor '{name}' instantiated a forbidden dummy host!"
        except TypeError as exc:
            # Clear error requiring model is completely valid
            assert "model" in str(exc).lower() or "signature" in str(exc).lower()
