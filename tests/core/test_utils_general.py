#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.utils.general."""

from __future__ import annotations

import random
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from neuralops.core.utils.general import (
    clear_memory,
    count_parameters,
    format_number,
    get_device,
    get_memory_usage,
    get_model_device,
    human_readable_time,
    set_seed,
    ste_clamp,
    ste_round,
)


def test_get_device_cpu() -> None:
    device = get_device("cpu")
    assert device == torch.device("cpu")


def test_get_device_auto() -> None:
    device = get_device("auto")
    assert isinstance(device, torch.device)
    if torch.cuda.is_available():
        assert device.type == "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        assert device.type == "mps"
    else:
        assert device.type == "cpu"


def test_get_device_when_device_unavailable() -> None:
    with patch("torch.cuda.is_available", return_value=False):
        with pytest.raises(RuntimeError, match="CUDA was requested, but no CUDA device is available"):
            get_device("cuda")


def test_get_device_index_out_of_range() -> None:
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.cuda.device_count", return_value=2):
            with pytest.raises(RuntimeError, match="CUDA device index 4 is unavailable"):
                get_device("cuda:4")


def test_set_seed_deterministic() -> None:
    set_seed(42, deterministic=True)
    val_py1 = random.random()
    val_np1 = np.random.rand()
    val_torch1 = torch.rand(3)

    set_seed(42, deterministic=True)
    val_py2 = random.random()
    val_np2 = np.random.rand()
    val_torch2 = torch.rand(3)

    assert val_py1 == val_py2
    assert val_np1 == val_np2
    assert torch.equal(val_torch1, val_torch2)


def test_get_model_device() -> None:
    model = nn.Linear(4, 4)
    assert get_model_device(model) == torch.device("cpu")

    empty_model = nn.Identity()
    assert get_model_device(empty_model) == torch.device("cpu")


def test_count_parameters() -> None:
    model = nn.Sequential(
        nn.Linear(10, 5, bias=False),  # 50 params
        nn.Linear(5, 2, bias=True),    # 10 + 2 = 12 params
    )
    # Total = 62
    assert count_parameters(model, trainable_only=False) == 62
    assert count_parameters(model, trainable_only=True) == 62

    # Freeze first layer
    for p in model[0].parameters():
        p.requires_grad = False

    assert count_parameters(model, trainable_only=True) == 12
    assert count_parameters(model, trainable_only=False) == 62


def test_format_number() -> None:
    assert format_number(500) == "500"
    assert format_number(1_500) == "2K" or format_number(1_500) == "1K"
    assert format_number(2_400_000) == "2.4M"


def test_clear_memory_and_get_memory_usage() -> None:
    clear_memory()
    usage = get_memory_usage()
    if torch.cuda.is_available():
        assert isinstance(usage, float)
    else:
        assert usage is None


def test_human_readable_time() -> None:
    assert human_readable_time(45) == "45s"
    assert human_readable_time(125) == "2m 5s"
    assert human_readable_time(3665) == "1h 1m 5s"


def test_ste_round() -> None:
    x = torch.tensor([1.2, 2.7, 3.5], requires_grad=True)
    out = ste_round(x)

    # Forward value should be rounded
    expected = torch.round(torch.tensor([1.2, 2.7, 3.5]))
    assert torch.allclose(out, expected)

    # Backward gradient should be 1.0 (straight-through)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))


def test_ste_clamp() -> None:
    x = torch.tensor([-2.0, 0.5, 4.0], requires_grad=True)
    out = ste_clamp(x, min_val=-1.0, max_val=2.0)

    # Forward value should be clamped
    expected = torch.tensor([-1.0, 0.5, 2.0])
    assert torch.allclose(out, expected)

    # Backward gradient should be 1.0 everywhere (straight-through)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.allclose(x.grad, torch.ones_like(x))
