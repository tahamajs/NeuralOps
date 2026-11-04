#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Integration and Unit Tests for Experiment Tracking & Hardware Telemetry Subsystem.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from neuralops.benchmarks.telemetry.hardware import (
    HardwareTelemetryMonitor,
    get_hardware_telemetry,
    log_hardware_telemetry,
)
from neuralops.benchmarks.telemetry.host import get_host_environment
from neuralops.benchmarks.telemetry.latency import LatencyDistributionTracker
from neuralops.benchmarks.telemetry.logging import ConsoleLogCapture
from neuralops.benchmarks.telemetry.manifest import (
    BenchmarkManifestBuilder,
    create_manifest,
    get_git_provenance,
)
from neuralops.benchmarks.telemetry.memory_forensics import GPUMemoryForensics
from neuralops.tracking.backends import LOCAL_BACKEND, TENSORBOARD_BACKEND, LocalTracker, TensorBoardTracker
from neuralops.tracking.tracker import ExperimentTracker


def test_experiment_tracker_local_init_and_metrics(tmp_path: Path):
    """Test LocalTracker initialization and metric streaming to metrics.jsonl."""
    tracker = ExperimentTracker(log_dir=tmp_path, backend="local")
    assert tracker.log_dir == str(tmp_path)

    tracker.log_metric("latency_ms", 12.5, step=0)
    tracker.log_metric("ppl", 16.0, step=1)
    tracker.log_params({"model": "gpt2", "batch_size": 4})
    tracker.finalize({"eval_loss": 2.718})

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists(), "metrics.jsonl must exist"

    lines = [json.loads(line) for line in metrics_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2
    assert lines[0]["name"] == "latency_ms"
    assert lines[0]["value"] == 12.5
    assert lines[0]["step"] == 0
    assert lines[1]["name"] == "ppl"
    assert lines[1]["value"] == 16.0
    assert lines[1]["step"] == 1

    params_file = tmp_path / "params.json"
    assert params_file.exists()
    params = json.loads(params_file.read_text(encoding="utf-8"))
    assert params["model"] == "gpt2"

    summary_file = tmp_path / "summary.json"
    assert summary_file.exists()
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    assert summary["eval_loss"] == 2.718


def test_experiment_tracker_tensorboard_directory_isolation(tmp_path: Path):
    """Test TensorBoardTracker creates directory under log_dir/tensorboard/."""
    tracker = ExperimentTracker(log_dir=tmp_path, backend="tensorboard")
    tb_dir = tmp_path / "tensorboard"
    assert tb_dir.exists(), "tensorboard/ directory must exist in log_dir"

    tracker.log_metric("loss", 0.45, step=0)
    tracker.close()


def test_multi_backend_dispatch(tmp_path: Path):
    """Test concurrent dispatch to both LocalTracker and TensorBoardTracker."""
    tracker = ExperimentTracker(log_dir=tmp_path, backend=["local", "tensorboard"])
    tracker.log_metric("throughput", 4200.0, step=0)
    tracker.close()

    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert "throughput" in lines[0]


def test_step_progression_and_batch_metrics(tmp_path: Path):
    """Test step advancement and batch metric logging."""
    with ExperimentTracker(log_dir=tmp_path) as tracker:
        tracker.set_step(10)
        assert tracker.get_step() == 10
        tracker.step(5)
        assert tracker.get_step() == 15

        tracker.log_metrics_batch({"metric_a": 1.0, "metric_b": 2.0})

    lines = (tmp_path / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    entries = [json.loads(l) for l in lines]
    assert all(e["step"] == 15 for e in entries)


def test_hardware_telemetry_sampling_and_fallback():
    """Verify hardware telemetry sampling returns valid finite numbers."""
    telemetry = get_hardware_telemetry()
    assert isinstance(telemetry, dict)
    for key in ("gpu_temp_c", "gpu_vram_used_mb", "gpu_power_w"):
        assert key in telemetry
        assert isinstance(telemetry[key], (int, float))
        assert math.isfinite(telemetry[key])
        assert not math.isnan(telemetry[key])


def test_hardware_telemetry_monitor(tmp_path: Path):
    """Verify on-demand telemetry logging to metrics.jsonl."""
    tracker = ExperimentTracker(log_dir=tmp_path)
    monitor = HardwareTelemetryMonitor(tracker=tracker)
    snapshot = monitor.log_step(step=0)
    assert "gpu_temp_c" in snapshot

    tracker.close()
    metrics_file = tmp_path / "metrics.jsonl"
    assert metrics_file.exists()
    lines = metrics_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 3


def test_host_environment_profiler():
    """Verify host platform inspection returns valid JSON-serializable dictionary."""
    env = get_host_environment()
    assert isinstance(env, dict)
    assert "hostname" in env
    assert "platform" in env
    assert "cpu_info" in env
    assert "memory_info" in env
    # Must be JSON-serializable
    serialized = json.dumps(env)
    assert len(serialized) > 50


def test_manifest_builder_and_git_provenance(tmp_path: Path):
    """Verify execution manifest builder adheres to Appendix B schema."""
    builder = BenchmarkManifestBuilder(run_id="run_test_123", benchmark_name="attention")
    builder.record_cell_result("PASSED")
    builder.record_cell_result("PASSED")
    builder.record_cell_result("FAILED_OOM")
    builder.record_outcome(status="COMPLETED", exit_code=0)

    manifest_file = builder.save(tmp_path)
    assert manifest_file.exists()

    data = json.loads(manifest_file.read_text(encoding="utf-8"))
    assert data["run_id"] == "run_test_123"
    assert data["benchmark_name"] == "attention"
    assert "git_provenance" in data
    assert "host_environment" in data
    assert data["execution_outcome"]["status"] == "COMPLETED"
    assert data["execution_outcome"]["passed_cells"] == 2
    assert data["execution_outcome"]["failed_oom_cells"] == 1


def test_console_log_capture(tmp_path: Path):
    """Verify ConsoleLogCapture mirrors stdout and stderr to stdout_stderr.log."""
    log_dir = tmp_path / "run_logs"
    with ConsoleLogCapture(log_dir):
        print("Hello from test stdout")

    log_file = log_dir / "stdout_stderr.log"
    assert log_file.exists()
    content = log_file.read_text(encoding="utf-8")
    assert "Hello from test stdout" in content


def test_latency_distribution_tracker():
    """Verify streaming latency percentiles against moments."""
    tracker = LatencyDistributionTracker()
    for val in [10.0, 20.0, 30.0, 40.0, 50.0]:
        tracker.record(val)

    summary = tracker.summary()
    assert summary["count"] == 5
    assert summary["mean_ms"] == 30.0
    assert summary["min_ms"] == 10.0
    assert summary["max_ms"] == 50.0
    assert summary["p50_ms"] == 30.0


def test_gpu_memory_forensics(tmp_path: Path):
    """Verify memory leak tracking and OOM diagnostic handler."""
    forensics = GPUMemoryForensics(snapshot_dir=tmp_path)
    step_info = forensics.step(0)
    assert "allocated_mb" in step_info

    oom_info = forensics.handle_oom(RuntimeError("CUDA out of memory"), step_idx=1)
    assert oom_info["status"] == "FAILED_OOM"
    assert "CUDA out of memory" in oom_info["error_message"]


def test_context_manager_and_idempotent_close(tmp_path: Path):
    """Verify context manager exit guarantees and idempotent close calls."""
    with ExperimentTracker(log_dir=tmp_path) as tracker:
        tracker.log_metric("step_loss", 0.1, step=1)

    # Calling close multiple times must be safe
    tracker.close()
    tracker.close()
