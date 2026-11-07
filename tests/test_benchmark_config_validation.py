#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for BenchmarkConfig strict validation (finding F-94).
"""

from __future__ import annotations

import pytest
from neuralops.benchmarks.config import BenchmarkConfig


def test_unknown_key_warns_or_fails():
    """Verify BenchmarkConfig.from_dict rejects unknown keys instead of silently burying them."""
    data = {
        "model": "distilbert-base-uncased",
        "dataset": "sst2",
        "unknown_typo_key": "some_value",
    }
    with pytest.raises(ValueError, match="Unknown configuration key"):
        BenchmarkConfig.from_dict(data)


def test_invalid_epochs_lr_rank_device_timeout_fail():
    """Verify BenchmarkConfig.__post_init__ rejects invalid values with ValueError."""
    # Invalid epochs
    with pytest.raises(ValueError, match="epochs"):
        BenchmarkConfig(epochs=0)
    with pytest.raises(ValueError, match="epochs"):
        BenchmarkConfig(epochs=-1)

    # Invalid lr
    with pytest.raises(ValueError, match="lr"):
        BenchmarkConfig(lr=0.0)
    with pytest.raises(ValueError, match="lr"):
        BenchmarkConfig(lr=-1e-4)

    # Invalid rank
    with pytest.raises(ValueError, match="rank"):
        BenchmarkConfig(rank=0)
    with pytest.raises(ValueError, match="rank"):
        BenchmarkConfig(rank=-16)

    # Invalid device
    with pytest.raises(ValueError, match="Invalid device"):
        BenchmarkConfig(device="tpu")
    with pytest.raises(ValueError, match="Invalid device"):
        BenchmarkConfig(device="cuda:invalid")

    # Invalid timeout_s
    with pytest.raises(ValueError, match="timeout_s"):
        BenchmarkConfig(timeout_s=0)
    with pytest.raises(ValueError, match="timeout_s"):
        BenchmarkConfig(timeout_s=-100)
