#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for the canonical benchmark configuration contracts."""

from pathlib import Path
import json

import pytest

from neuralops.benchmarks.config import BenchmarkConfig as CanonicalConfig
from neuralops.benchmarks.runner import BaseRunner
from scripts.benchmarks.benchmark_config import BenchmarkConfig as LegacyConfig


def test_legacy_config_adapts_to_canonical_config(tmp_path: Path):
    legacy = LegacyConfig(device="cpu", output_dir=tmp_path)
    canonical = legacy.to_canonical()

    assert isinstance(canonical, CanonicalConfig)
    assert canonical.device == "cpu"
    assert canonical.output_dir == tmp_path
    assert canonical.extra["enable_backward"] is True


def test_canonical_config_roundtrip(tmp_path: Path):
    path = tmp_path / "config.json"
    original = CanonicalConfig(device="cpu", output_dir=tmp_path, seed=123)
    path.write_text(json.dumps(original.to_dict()), encoding="utf-8")
    restored = CanonicalConfig.from_dict(json.loads(path.read_text()))

    assert restored.to_dict() == original.to_dict()


def test_cuda_runner_verification_is_fail_closed(monkeypatch):
    class ProbeRunner(BaseRunner):
        def run(self):
            return []

    runner = ProbeRunner(CanonicalConfig(device="cuda"))
    monkeypatch.setattr("neuralops.benchmarks.runner.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        runner.verify()
