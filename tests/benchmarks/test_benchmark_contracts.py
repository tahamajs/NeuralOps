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
    repo_root = Path(__file__).resolve().parent.parent.parent
    config_path = repo_root / "benchmarks" / "configs" / "quick_cuda.json"
    config = CanonicalConfig.from_file(config_path)

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


def test_unsupported_precision_raises_error():
    """Bug 18: Unsupported precision names must raise ValueError immediately."""
    with pytest.raises(ValueError, match="Unsupported precision 'invalid_fp'"):
        CanonicalConfig(precision="invalid_fp")


def test_supported_precision_aliases_normalize():
    """Bug 18: Precision aliases (fp16, bf16, float32) must normalize consistently."""
    cfg_fp16 = CanonicalConfig(precision="fp16")
    assert cfg_fp16.precision == "float16"

    cfg_bf16 = CanonicalConfig(precision="bf16")
    assert cfg_bf16.precision == "bfloat16"

    cfg_fp32 = CanonicalConfig(precision="fp32")
    assert cfg_fp32.precision == "float32"


def test_parquet_missing_dependency_fails_honestly(tmp_path: Path, monkeypatch):
    """Bug 10: Optional Parquet failure must not produce a fake file or masquerade as saved."""
    from neuralops.benchmarks.results.saver import save_all_formats
    from neuralops.benchmarks.results.schema import BenchmarkResult

    result = BenchmarkResult()
    result.run_id = "test_parquet_run"
    result.module = {"name": "test_mod", "type": "kernel"}

    def _failing_parquet(*args, **kwargs):
        raise ImportError("No module named 'pyarrow'")

    monkeypatch.setattr("neuralops.benchmarks.results.saver.save_parquet_index", _failing_parquet)

    # When require_parquet=False: should warn and omit 'parquet' from saved dict
    saved = save_all_formats(result, output_dir=tmp_path, require_parquet=False)
    assert "parquet" not in saved
    assert not (tmp_path / "results.parquet").exists()

    # When require_parquet=True: should raise RuntimeError
    with pytest.raises(RuntimeError, match="Parquet export failed"):
        save_all_formats(result, output_dir=tmp_path, require_parquet=True)


def test_cli_config_preset_deep_override(tmp_path: Path):
    """Bug 7: CLI handle_run must deeply apply all fields from the config file."""
    import argparse
    from neuralops.benchmarks.cli import apply_config_to_args

    preset_path = tmp_path / "custom_preset.json"
    custom_config = CanonicalConfig(
        warmup=9,
        iters=42,
        seed=999,
        precision="float16",
        device="cpu",
        output_dir=tmp_path / "custom_out",
    )
    preset_path.write_text(json.dumps(custom_config.to_dict()), encoding="utf-8")

    args = argparse.Namespace(
        config=str(preset_path),
        device="cpu",
        warmup=3,
        iters=10,
        seed=42,
        precision="float32",
        output=None,
        compare_with=None,
        modules=None,
        category=None,
        verify_only=False,
    )

    apply_config_to_args(args)
    assert args.warmup == 9
    assert args.iters == 42
    assert args.seed == 999
    assert args.precision == "float16"
    assert args.device == "cpu"
    assert args.output_dir == str(tmp_path / "custom_out")
