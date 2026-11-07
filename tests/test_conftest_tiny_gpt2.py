#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying tiny_gpt2 fixture fail-loud behavior without silent dummy fallback."""

import sys
from unittest.mock import patch
import pytest
import torch


def test_tiny_gpt2_dummy_fixture_explicit(tiny_gpt2_dummy):
    """Verify tiny_gpt2_dummy fixture returns a working mock model when explicitly requested."""
    assert tiny_gpt2_dummy is not None
    dummy_input = torch.tensor([[1, 2, 3]])
    out = tiny_gpt2_dummy(dummy_input)
    assert hasattr(out, "logits")
    assert out.logits.shape[-1] == 256


def test_tiny_gpt2_fails_when_transformers_missing(monkeypatch):
    """Verify tiny_gpt2 fixture fails loudly with Skip/ImportError when transformers is missing."""
    import builtins
    real_import = builtins.__import__

    def mock_import(name, *args, **kwargs):
        if name == "transformers" or name.startswith("transformers."):
            raise ModuleNotFoundError("No module named 'transformers'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    from tests.conftest import tiny_gpt2

    # Should raise pytest.skip (Skipped exception) or ImportError, never return a dummy
    fn = getattr(tiny_gpt2, "__wrapped__", tiny_gpt2)
    with pytest.raises((pytest.skip.Exception, ImportError, ModuleNotFoundError)):
        fn()
