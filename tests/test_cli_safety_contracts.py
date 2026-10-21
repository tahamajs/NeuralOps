#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for lazy CLI imports and honest failure behavior."""

from __future__ import annotations

import argparse
import subprocess
import sys
import types

import pytest
import torch.nn as nn


def test_attention_and_compression_package_imports_are_lazy() -> None:
    script = (
        "import sys; "
        "import neuralops.attentions; "
        "assert 'neuralops.attentions.linear' not in sys.modules; "
        "import neuralops.compression_engine; "
        "assert 'neuralops.compression_engine.svd' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_benchmark_cli_import_does_not_import_torch() -> None:
    script = (
        "import sys; import neuralops.benchmarks.cli as cli; cli._build_parser(); "
        "assert 'torch' not in sys.modules"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr


def test_compression_checkpoint_load_failure_does_not_make_placeholder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuralops.cli.compress_cmd import _load_model_for_compression

    def fail_load(*args: object, **kwargs: object) -> None:
        raise OSError("checkpoint unavailable")

    transformers = types.ModuleType("transformers")
    transformers.AutoModel = types.SimpleNamespace(from_pretrained=fail_load)
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fail_load)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(RuntimeError, match="No placeholder model was created"):
        _load_model_for_compression("missing-checkpoint", "cpu")


def test_compressor_failure_does_not_substitute_another_algorithm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    import neuralops
    import neuralops.cli.compress_cmd as compress_cmd

    monkeypatch.setattr(compress_cmd, "_load_model_for_compression", lambda *_args, **_kwargs: nn.Linear(4, 4))

    def fail_compress(**_kwargs: object) -> None:
        raise ValueError("requested method failed")

    monkeypatch.setattr(neuralops, "compress", fail_compress)
    output_dir = tmp_path / "must-not-exist"
    args = argparse.Namespace(
        model="fixture-model",
        compressor="direct_svd",
        rank=2,
        target_ratio=None,
        output_dir=str(output_dir),
        device="cpu",
        json=True,
    )

    with pytest.raises(RuntimeError, match="no alternate compressor was substituted"):
        compress_cmd.cmd_compress(args)
    assert not output_dir.exists()


def test_serving_checkpoint_load_failure_does_not_make_random_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from neuralops.cli.serve_cmd import _load_serving_model

    def fail_load(*args: object, **kwargs: object) -> None:
        raise OSError("checkpoint unavailable")

    transformers = types.ModuleType("transformers")
    transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=fail_load)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    with pytest.raises(RuntimeError, match="will not start with random placeholder weights"):
        _load_serving_model("missing-checkpoint", "cpu")


def test_serve_parser_rejects_unimplemented_engines() -> None:
    from neuralops.benchmarks.cli import _build_parser

    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["serve", "--engine", "continuous_batcher"])
    args = parser.parse_args(["serve", "--model", "gpt2", "--port", "8123"])
    assert args.model == "gpt2"
    assert args.port == 8123
    assert args.engine == "fastapi"
