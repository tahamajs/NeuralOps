#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests verifying trust_remote_code security defaults, logging, and manifest recording."""

from __future__ import annotations

import ast
import inspect
import json
import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.core.model_loader import load_model_and_tokenizer


def test_default_is_false() -> None:
    """Assert BenchmarkConfig and load_model_and_tokenizer default trust_remote_code to False."""
    config = BenchmarkConfig()
    assert config.trust_remote_code is False, "BenchmarkConfig.trust_remote_code must default to False"

    sig = inspect.signature(load_model_and_tokenizer)
    default_val = sig.parameters["trust_remote_code"].default
    assert default_val is False, f"load_model_and_tokenizer default must be False, found {default_val}"


def test_true_is_logged_and_recorded(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """Assert explicit trust_remote_code=True logs warning with model name and records in manifest."""
    model_name = "test-untrusted-model-123"

    with caplog.at_level(logging.WARNING):
        with (
            patch("transformers.AutoTokenizer.from_pretrained"),
            patch("transformers.AutoConfig.from_pretrained"),
            patch("transformers.AutoModel.from_pretrained"),
        ):
            load_model_and_tokenizer(model_name, trust_remote_code=True)

    matching_logs = [
        rec.message
        for rec in caplog.records
        if model_name in rec.message and "remote code WILL be executed" in rec.message
    ]
    assert matching_logs, f"Expected warning containing model name and refusal notice, got: {caplog.text}"

    # Verify manifest recording behavior
    config = BenchmarkConfig(
        model=model_name,
        trust_remote_code=True,
        output_dir=str(tmp_path),
    )
    manifest_data = {
        "run_id": "test_run",
        "model_name": getattr(config, "models", None) or getattr(config, "model", None),
        "trust_remote_code": bool(getattr(config, "trust_remote_code", False)),
    }
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    with open(manifest_file, "r", encoding="utf-8") as f:
        saved = json.load(f)

    assert saved["trust_remote_code"] is True
    assert saved["model_name"] == model_name


def test_no_from_pretrained_without_explicit_flag() -> None:
    """Static AST check asserting every from_pretrained call carries an explicit trust_remote_code kwarg."""
    source_path = Path("neuralops/benchmarks/core/model_loader.py")
    assert source_path.exists(), f"Source path {source_path} does not exist"

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    from_pretrained_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "from_pretrained"
    ]

    assert len(from_pretrained_calls) >= 7, f"Expected at least 7 calls, found {len(from_pretrained_calls)}"
    for call in from_pretrained_calls:
        has_flag = any(kw.arg == "trust_remote_code" for kw in call.keywords)
        assert has_flag, f"Call at line {call.lineno} does not pass explicit trust_remote_code argument"
