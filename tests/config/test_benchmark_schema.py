#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for Pydantic V2 benchmark schema models in neuralops.config.benchmark_schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from neuralops.config.benchmark_schema import (
    AttentionSweepParameters,
    BenchmarkRunConfig,
    CompressionSweepParameters,
    HardwareExecutionTarget,
    TrackingSubsystemConfig,
)


class TestHardwareExecutionTarget:
    """Verifies HardwareExecutionTarget schema defaults and validation."""

    def test_defaults_are_valid(self):
        target = HardwareExecutionTarget()
        assert target.device == "cuda:0"
        assert target.backend_preference == "auto"
        assert target.enforce_deterministic is False
        assert target.allow_tf32 is True
        assert target.target_compute_capability is None
        assert target.memory_fraction_limit is None

    def test_device_validation(self):
        # Valid device strings
        assert HardwareExecutionTarget(device="cpu").device == "cpu"
        assert HardwareExecutionTarget(device="mps").device == "mps"
        assert HardwareExecutionTarget(device="cuda").device == "cuda"
        assert HardwareExecutionTarget(device="cuda:1").device == "cuda:1"
        assert HardwareExecutionTarget(device="0").device == "cuda:0"
        assert HardwareExecutionTarget(device="2").device == "cuda:2"
        assert HardwareExecutionTarget(device="auto").device == "auto"

        # Invalid device string
        with pytest.raises(ValidationError):
            HardwareExecutionTarget(device="tpu")

        with pytest.raises(ValidationError):
            HardwareExecutionTarget(device="invalid_dev")


class TestTrackingSubsystemConfig:
    """Verifies TrackingSubsystemConfig schema defaults and validation."""

    def test_defaults_are_valid(self):
        tracking = TrackingSubsystemConfig()
        assert tracking.enabled is True
        assert tracking.enable_tensorboard is True
        assert tracking.enable_local_jsonl is True
        assert tracking.log_frequency_steps == 1
        assert tracking.capture_gpu_telemetry is True
        assert tracking.telemetry_sample_interval_ms == 250
        assert tracking.tensorboard_flush_secs == 10
        assert tracking.export_formats == ["jsonl", "csv", "summary"]

    def test_bounds_validation(self):
        # log_frequency_steps >= 1
        with pytest.raises(ValidationError):
            TrackingSubsystemConfig(log_frequency_steps=0)

        # telemetry_sample_interval_ms ge=50, le=10000
        with pytest.raises(ValidationError):
            TrackingSubsystemConfig(telemetry_sample_interval_ms=10)

        with pytest.raises(ValidationError):
            TrackingSubsystemConfig(telemetry_sample_interval_ms=20000)


class TestAttentionSweepParameters:
    """Verifies AttentionSweepParameters schema defaults and validation."""

    def test_defaults_are_valid(self):
        params = AttentionSweepParameters()
        assert params.mechanisms == ["all"]
        assert params.batch_sizes == [1, 4, 16]
        assert params.sequence_lengths == [512, 1024, 2048, 4096, 8192]
        assert params.head_dimensions == [64, 128]
        assert params.num_heads == 32
        assert params.num_kv_heads is None
        assert params.dtypes == ["fp16", "bf16"]
        assert params.causal_modes == [True, False]
        assert params.warmup_iterations == 10
        assert params.benchmark_iterations == 50

    def test_dimension_validation(self):
        # Empty list is disallowed
        with pytest.raises(ValidationError):
            AttentionSweepParameters(batch_sizes=[])

        # Non-positive values disallowed
        with pytest.raises(ValidationError):
            AttentionSweepParameters(batch_sizes=[0, 4])

        with pytest.raises(ValidationError):
            AttentionSweepParameters(sequence_lengths=[-1, 512])

        # Automatic deduplication and sorting
        params = AttentionSweepParameters(batch_sizes=[16, 4, 1, 4])
        assert params.batch_sizes == [1, 4, 16]


class TestCompressionSweepParameters:
    """Verifies CompressionSweepParameters schema defaults and validation."""

    def test_defaults_are_valid(self):
        params = CompressionSweepParameters()
        assert params.compressors == ["all"]
        assert params.target_layers == ["all"]
        assert params.target_ratios == [0.25, 0.50, 0.75]
        assert params.evaluation_datasets == ["wikitext2"]
        assert params.eval_context_length == 2048

    def test_ratio_validation(self):
        # Target ratios must be strictly between 0 and 1
        with pytest.raises(ValidationError):
            CompressionSweepParameters(target_ratios=[0.0, 0.5])

        with pytest.raises(ValidationError):
            CompressionSweepParameters(target_ratios=[0.5, 1.0])

        with pytest.raises(ValidationError):
            CompressionSweepParameters(target_ratios=[1.5])

        # Deduplication and sorting
        params = CompressionSweepParameters(target_ratios=[0.75, 0.25, 0.50, 0.25])
        assert params.target_ratios == [0.25, 0.50, 0.75]


class TestBenchmarkRunConfig:
    """Verifies BenchmarkRunConfig canonical configuration model."""

    def test_defaults_are_valid(self):
        config = BenchmarkRunConfig(benchmark_suite="attention")
        assert config.version == "1.0.0"
        assert config.benchmark_suite == "attention"
        assert config.name is None
        assert config.hardware.device == "cuda:0"
        assert config.tracking.enabled is True
        assert config.attention_sweep is None
        assert config.compression_sweep is None
        assert config.custom_parameters == {}

    def test_suite_validation(self):
        # Valid suites
        valid_suites = [
            "attention",
            "compression",
            "cuda_ops",
            "triton_ops",
            "e2e_serving",
            "niah",
            "roofline",
            "qwen_kd",
        ]
        for suite in valid_suites:
            cfg = BenchmarkRunConfig(benchmark_suite=suite)
            assert cfg.benchmark_suite == suite

        # Invalid suite raises ValidationError
        with pytest.raises(ValidationError):
            BenchmarkRunConfig(benchmark_suite="unsupported_suite")

    def test_yaml_roundtrip(self, tmp_path):
        config = BenchmarkRunConfig(
            benchmark_suite="attention",
            name="ci_test_run",
            attention_sweep=AttentionSweepParameters(batch_sizes=[2, 8]),
        )
        yaml_str = config.model_dump_yaml()
        loaded = BenchmarkRunConfig.from_yaml(yaml_str)
        assert loaded.benchmark_suite == "attention"
        assert loaded.name == "ci_test_run"
        assert loaded.attention_sweep is not None
        assert loaded.attention_sweep.batch_sizes == [2, 8]

        # File roundtrip
        yaml_file = tmp_path / "benchmark_run.yaml"
        yaml_file.write_text(yaml_str, encoding="utf-8")
        loaded_file = BenchmarkRunConfig.from_yaml(yaml_file)
        assert loaded_file.name == "ci_test_run"
        assert loaded_file.attention_sweep.batch_sizes == [2, 8]
