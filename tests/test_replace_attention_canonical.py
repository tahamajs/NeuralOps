#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for canonical replace_attention single source of truth (finding F-92).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import neuralops.surgery as surgery
from neuralops.api import replace_attention as r_api
from neuralops.core._attention_surgery import replace_attention as r_core_surgery
from neuralops.core.attention_utils import replace_attention as r_core_utils
from neuralops.operations.attention.replace import AttentionReplaceOperation
from neuralops.surgery.attention.replace import replace_attention as r_surg_attn
from neuralops.utils.replace_attention import replace_attention as r_utils
from examples._shared.mock_models import TinyClassifier


def test_all_replace_paths_are_same_function():
    """Verify that all replace_attention paths point to the identical canonical object."""
    assert r_api is r_core_utils
    assert r_api is r_core_surgery
    assert r_api is r_surg_attn
    assert r_api is r_utils


def test_surgery_all_lists_replace_once():
    """Verify that surgery.__all__ contains 'replace_attention' exactly once (no shadowing)."""
    assert surgery.__all__.count("replace_attention") == 1


def test_operations_replace_end_to_end():
    """Verify end-to-end execution of AttentionReplaceOperation using canonical keyword."""
    model = TinyClassifier()
    op = AttentionReplaceOperation()
    transformed, meta = op.apply(model, {"target": "cosformer"})
    assert transformed is not None
    assert meta["params_used"]["target"] == "cosformer"
    assert meta["output_state"] == "attention_replaced"
