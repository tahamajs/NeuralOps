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
from neuralops.benchmarks.regression_gate import compare_results
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


def test_canonical_config_loads_accelerator_preset():
    config = CanonicalConfig.from_file(Path("benchmarks/configs/quick_cuda.json"))

    assert config.device == "cuda"
    assert config.precision == "float32"
    assert config.batch_sizes == [1, 2]
    assert config.seq_lens == [128, 512]


def test_requested_accelerator_is_fail_closed(monkeypatch):
    class ProbeRunner(BaseRunner):
        def run(self):
            return []

    runner = ProbeRunner(CanonicalConfig(device="cuda"))
    monkeypatch.setattr("neuralops.benchmarks.runner.torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        runner.verify()


def test_regression_gate_requires_real_metrics(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    baseline.write_text('{"run_id":"base","metrics":{"p95_ms":10.0}}', encoding="utf-8")
    candidate.write_text('{"run_id":"candidate","metrics":{"p95_ms":11.0}}', encoding="utf-8")

    result = compare_results(baseline, candidate, "p95_ms", max_regression=2.0)
    assert result["status"] == "PASS"
    assert result["regression"] == 1.0

    with pytest.raises(KeyError):
        compare_results(baseline, candidate, "throughput", max_regression=2.0)
