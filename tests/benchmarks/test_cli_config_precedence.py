#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for deterministic config precedence and environment variable merging."""

import argparse
import json
from pathlib import Path

import pytest

from neuralops.cli.core.config_merge import load_effective_config
from neuralops.cli.core.errors import ConfigurationError


def test_config_precedence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg_file = tmp_path / "test_cfg.json"
    cfg_file.write_text(json.dumps({
        "warmup": 10,
        "iters": 50,
        "batch_sizes": [4],
        "device": "cpu",
    }), encoding="utf-8")

    monkeypatch.setenv("NEURALOPS_DEVICE", "cpu")
    monkeypatch.setenv("NEURALOPS_PRECISION", "bfloat16")

    # CLI explicitly provides iters=100
    args = argparse.Namespace(
        config=str(cfg_file),
        iters=100,
        warmup=None,
        device=None,
        precision=None,
        dtype=None,
        output=None,
        batch_size=None,
        seq_len=None,
    )

    effective = load_effective_config(args)

    # CLI flag overrides everything
    assert effective.iters == 100
    # Env var overrides config file
    assert effective.precision == "bfloat16"
    assert effective.device == "cpu"
    # Config file overrides built-in default
    assert effective.warmup == 10
    assert effective.batch_sizes == [4]


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    cfg_file = tmp_path / "invalid_cfg.json"
    cfg_file.write_text(json.dumps({
        "unknown_invented_flag": 123,
    }), encoding="utf-8")

    args = argparse.Namespace(config=str(cfg_file))
    with pytest.raises(ConfigurationError, match="Unknown configuration key"):
        load_effective_config(args)
