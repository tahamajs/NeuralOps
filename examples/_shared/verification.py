#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Verification and parity checking utilities for NeuralOps examples.
"""

from __future__ import annotations

import importlib
from typing import Any

import torch
import torch.nn as nn

from examples._shared.report import pass_fail, skip


def assert_parity(
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float = 1e-4,
    rtol: float = 1e-3,
    label: str = "Parity Check",
) -> bool:
    """Verify numerical parity between two tensors and report PASS/FAIL."""
    try:
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        pass_fail(label, True, f"atol={atol}, rtol={rtol}")
        return True
    except AssertionError as err:
        max_diff = (actual - expected).abs().max().item()
        pass_fail(label, False, f"max_diff={max_diff:.6e} > atol={atol}")
        return False


def assert_shape_unchanged(
    model_before: nn.Module,
    model_after: nn.Module,
    label: str = "Parameter Shape Invariant",
) -> bool:
    """Verify that all parameter names and shapes match before and after an operation."""
    dict_before = {name: param.shape for name, param in model_before.named_parameters()}
    dict_after = {name: param.shape for name, param in model_after.named_parameters()}

    if dict_before == dict_after:
        pass_fail(label, True, f"{len(dict_before)} parameters preserved")
        return True

    mismatches = []
    for name, shape in dict_before.items():
        if name not in dict_after:
            mismatches.append(f"Missing '{name}'")
        elif dict_after[name] != shape:
            mismatches.append(f"Shape '{name}': {shape} -> {dict_after[name]}")
    for name in dict_after:
        if name not in dict_before:
            mismatches.append(f"Unexpected new parameter '{name}'")

    pass_fail(label, False, "; ".join(mismatches[:3]))
    return False


def skip_if_missing(module_name: str, reason: str | None = None) -> bool:
    """Check if an optional module is importable; if missing, log skip and return True."""
    try:
        importlib.import_module(module_name)
        return False
    except ImportError:
        desc = reason or f"optional package '{module_name}' is not installed"
        skip(module_name, desc)
        return True
