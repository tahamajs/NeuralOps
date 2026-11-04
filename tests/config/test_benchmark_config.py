#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit test suite for NeuralOps benchmark configuration schemas and loader.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from pydantic import ValidationError

from neuralops.config.benchmark_schema import (
    AttentionSweepParameters,
    BenchmarkRunConfig,
    CompressionSweepParameters,
    HardwareExecutionTarget,
    TrackingSubsystemConfig,
)
from neuralops.config.loader import (
    _interpolate_env_vars,
    load_benchmark_config,
)


TEMPLATE_DIR = Path("neuralops/config/templates/benchmarks")


def test_all_yaml_templates_validate() -> None:
    """Every template in neuralops/config/templates/benchmarks/*.yaml must parse cleanly."""
    yaml_files = list(TEMPLATE_DIR.glob("*.yaml"))
    assert len(yaml_files) >= 3, f"Expected at least 3 templates, found {len(yaml_files)}"
    for yaml_path in yaml_files:
        config = load_benchmark_config(yaml_path)
        assert isinstance(config, BenchmarkRunConfig)
        assert config.version == "1.0.0"
        assert config.benchmark_suite in [
            "attention",
            "compression",
            "cuda_ops",
            "triton_ops",
            "e2e_serving",
            "niah",
            "roofline",
            "qwen_kd",
        ]
        dump = config.model_dump()
        assert isinstance(dump, dict)
        assert dump["benchmark_suite"] == config.benchmark_suite


def test_load_from_dict() -> None:
    """Direct dictionary source loading."""
    data = {
        "benchmark_suite": "attention",
        "name": "dict_test",
        "hardware": {"device": "cpu"},
    }
    cfg = load_benchmark_config(data)
    assert cfg.benchmark_suite == "attention"
    assert cfg.name == "dict_test"
    assert cfg.hardware.device == "cpu"


def test_load_from_json(tmp_path: Path) -> None:
    """JSON configuration loading."""
    json_path = tmp_path / "test_config.json"
    data = {
        "version": "1.0.0",
        "benchmark_suite": "compression",
        "hardware": {"device": "cuda:1"},
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    cfg = load_benchmark_config(json_path)
    assert cfg.benchmark_suite == "compression"
    assert cfg.hardware.device == "cuda:1"


def test_env_var_interpolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Environment variable substitution with and without defaults."""
    monkeypatch.setenv("TEST_CUDA_DEV", "cuda:2")
    raw = {
        "device_with_env": "${TEST_CUDA_DEV:-cpu}",
        "device_default": "${NON_EXISTENT_VAR:-cuda:3}",
        "device_empty": "${EMPTY_VAR}",
        "nested": {
            "val": "${TEST_CUDA_DEV:-cpu}",
            "list": ["${TEST_CUDA_DEV}", "${FALLBACK:-fixed}"],
        },
    }
    interpolated = _interpolate_env_vars(raw)
    assert interpolated["device_with_env"] == "cuda:2"
    assert interpolated["device_default"] == "cuda:3"
    assert interpolated["device_empty"] == ""
    assert interpolated["nested"]["val"] == "cuda:2"
    assert interpolated["nested"]["list"] == ["cuda:2", "fixed"]


def test_cli_overrides_nested_and_dotted() -> None:
    """CLI overrides support both nested dictionaries and dotted string keys."""
    data = {
        "benchmark_suite": "attention",
        "hardware": {"device": "cuda:0", "backend_preference": "auto"},
        "tracking": {"enabled": True},
    }
    # Dotted override
    cfg1 = load_benchmark_config(data, overrides={"hardware.device": "cpu"})
    assert cfg1.hardware.device == "cpu"
    assert cfg1.hardware.backend_preference == "auto"

    # Nested override
    cfg2 = load_benchmark_config(data, overrides={"hardware": {"device": "mps"}})
    assert cfg2.hardware.device == "mps"


def test_hardware_device_validation() -> None:
    """Hardware target validates device strings and numeric normalization."""
    hw0 = HardwareExecutionTarget(device="0")
    assert hw0.device == "cuda:0"

    hw_cuda = HardwareExecutionTarget(device="cuda:3")
    assert hw_cuda.device == "cuda:3"

    hw_cpu = HardwareExecutionTarget(device="cpu")
    assert hw_cpu.device == "cpu"

    hw_mps = HardwareExecutionTarget(device="mps")
    assert hw_mps.device == "mps"

    with pytest.raises(ValidationError):
        HardwareExecutionTarget(device="invalid_device:99")


def test_hardware_memory_fraction_validation() -> None:
    """Memory fraction limit must be 0 < x <= 1.0."""
    hw = HardwareExecutionTarget(memory_fraction_limit=0.5)
    assert hw.memory_fraction_limit == 0.5

    with pytest.raises(ValidationError):
        HardwareExecutionTarget(memory_fraction_limit=0.0)

    with pytest.raises(ValidationError):
        HardwareExecutionTarget(memory_fraction_limit=1.5)


def test_attention_sweep_parameters_validation() -> None:
    """Attention sweep checks positive integers and deduplicates."""
    params = AttentionSweepParameters(
        batch_sizes=[4, 1, 4, 16],
        sequence_lengths=[1024, 512, 1024],
        head_dimensions=[128, 64],
    )
    assert params.batch_sizes == [1, 4, 16]
    assert params.sequence_lengths == [512, 1024]
    assert params.head_dimensions == [64, 128]

    with pytest.raises(ValidationError):
        AttentionSweepParameters(batch_sizes=[0, 4])

    with pytest.raises(ValidationError):
        AttentionSweepParameters(sequence_lengths=[-512])


def test_compression_sweep_parameters_validation() -> None:
    """Compression sweep target ratios must be strictly between 0 and 1."""
    params = CompressionSweepParameters(target_ratios=[0.75, 0.25, 0.5])
    assert params.target_ratios == [0.25, 0.5, 0.75]

    with pytest.raises(ValidationError):
        CompressionSweepParameters(target_ratios=[0.0, 0.5])

    with pytest.raises(ValidationError):
        CompressionSweepParameters(target_ratios=[1.0, 0.5])


def test_invalid_suite_enum_rejected() -> None:
    """Benchmark suite rejects unregistered suite strings."""
    with pytest.raises(ValidationError):
        BenchmarkRunConfig(benchmark_suite="non_existent_suite")  # type: ignore[arg-type]


def test_tracking_subsystem_bounds() -> None:
    """Tracking subsystem bounds checks."""
    t = TrackingSubsystemConfig(telemetry_sample_interval_ms=100)
    assert t.telemetry_sample_interval_ms == 100

    with pytest.raises(ValidationError):
        TrackingSubsystemConfig(telemetry_sample_interval_ms=20)  # below 50

    with pytest.raises(ValidationError):
        TrackingSubsystemConfig(log_frequency_steps=0)  # ge=1


def test_yaml_dump_fidelity() -> None:
    """BenchmarkRunConfig can export YAML representation."""
    cfg = load_benchmark_config("neuralops/config/templates/benchmarks/attention.yaml")
    yaml_str = cfg.model_dump_yaml()
    assert "benchmark_suite: attention" in yaml_str
    assert "version: 1.0.0" in yaml_str


def test_missing_file_raises_not_found() -> None:
    """Missing config file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_benchmark_config("non_existent_config.yaml")


def test_unsupported_file_extension(tmp_path: Path) -> None:
    """Unsupported extension raises ValueError."""
    txt_path = tmp_path / "config.txt"
    txt_path.write_text("benchmark_suite: attention")
    with pytest.raises(ValueError, match="Unsupported configuration extension"):
        load_benchmark_config(txt_path)
