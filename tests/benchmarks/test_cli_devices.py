#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for device normalization and fail-closed logic."""

import pytest

from neuralops.cli.core.devices import resolve_device
from neuralops.cli.core.errors import BackendUnavailableError


def test_device_cpu_resolution() -> None:
    dev = resolve_device("cpu")
    assert dev.resolved == "cpu"
    assert dev.kind == "cpu"


def test_device_auto_resolution() -> None:
    dev = resolve_device("auto")
    assert dev.resolved in ("cpu", "cuda:0", "mps")


def test_device_fail_closed_when_accelerator_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(BackendUnavailableError, match="CUDA device 'cuda:0' was requested"):
        resolve_device("cuda")

    with pytest.raises(BackendUnavailableError, match="CUDA device 'cuda:0' was requested"):
        resolve_device("cuda:0")


def test_device_invalid_format() -> None:
    with pytest.raises(ValueError, match="Unsupported device string"):
        resolve_device("unknown_accelerator")
