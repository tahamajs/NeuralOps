#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests verifying HF token secret redaction and CLI argument refusal."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from neuralops.benchmarks.cli import _build_parser
from neuralops.benchmarks.config import BenchmarkConfig


def test_to_dict_excludes_token():
    """Assert to_dict and repr do not expose plaintext HF token."""
    secret = "hf_xxx_secret"
    cfg = BenchmarkConfig(hf_token=secret)
    serialized = json.dumps(cfg.to_dict())
    assert secret not in serialized, f"Plaintext secret {secret!r} found in to_dict() serialization"
    assert secret not in repr(cfg), f"Plaintext secret {secret!r} found in repr(config)"


def test_save_excludes_token(tmp_path: Path):
    """Assert config.save() writes JSON without exposing plaintext HF token."""
    secret = "hf_xxx_secret"
    cfg = BenchmarkConfig(hf_token=secret)
    save_file = tmp_path / "benchmark_config.json"
    cfg.save(save_file)

    saved_text = save_file.read_text(encoding="utf-8")
    assert secret not in saved_text, f"Plaintext secret {secret!r} leaked to saved JSON config file"


def test_no_hf_token_cli_flag():
    """Assert --hf-token is either not in the parser or warns and refuses the value."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["run", "--hf-token", "hf_test_secret"])
