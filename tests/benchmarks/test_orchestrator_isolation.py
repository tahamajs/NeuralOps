#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for benchmark subprocess isolation and watchdog runner (AGT-58 / Gap 5).

Verifies that:
  1. A worker crashing with SIGSEGV/abort does not kill the sweep; it is safely
     recorded as FAILED or HARDWARE_ERROR and the sweep continues.
  2. A worker that exceeds its timeout is forcefully killed, recorded as TIMEOUT,
     and the sweep continues.
  3. Failure tracebacks and metadata are persistently logged in events.jsonl.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from neuralops.benchmarks.executor import execute_single
from neuralops.benchmarks.orchestrator import BenchmarkOrchestrator
from neuralops.benchmarks.config import BenchmarkConfig


def test_crash_does_not_kill_sweep(tmp_path):
    """Simulate a worker SIGSEGV / hard abort (os._exit or os.abort) and assert the sweep continues."""
    events_file = tmp_path / "events.jsonl"

    with patch("neuralops.benchmarks.executor.discover_modules", return_value={
        "crash_module": MagicMock(type="attention", family="dense", supports_causal=True, supports_gqa=True, supports_rope=True, requires_cuda=False, timeout_s=10),
        "normal_module": MagicMock(type="attention", family="dense", supports_causal=True, supports_gqa=True, supports_rope=True, requires_cuda=False, timeout_s=10),
    }):
        # Run crash module with simulated crash
        res_crash = execute_single(
            module="crash_module",
            output_dir=str(tmp_path),
            timeout_s=5,
            _test_behavior="crash",
        )
        assert res_crash.status in ("hardware_error", "failed")
        assert "134" in (res_crash.error or "")

        # Run normal module directly after crash — prove process lived and can run next module
        res_normal = execute_single(
            module="normal_module",
            output_dir=str(tmp_path),
            timeout_s=5,
            _test_behavior="success",
        )
        assert res_normal.status == "success"

    # Verify events.jsonl captured both events
    assert events_file.exists()
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    assert len(events) >= 2
    assert any(e["module"] == "crash_module" and e["status"] in ("hardware_error", "failed") for e in events)
    assert any(e["module"] == "normal_module" and e["status"] == "success" for e in events)


def test_timeout_records_timeout(tmp_path):
    """A worker that sleeps past its timeout is terminated and recorded as TIMEOUT."""
    events_file = tmp_path / "events.jsonl"

    with patch("neuralops.benchmarks.executor.discover_modules", return_value={
        "sleep_module": MagicMock(type="attention", family="dense", supports_causal=True, supports_gqa=True, supports_rope=True, requires_cuda=False, timeout_s=1),
    }):
        # Short timeout of 1 second with sleeping behavior
        res = execute_single(
            module="sleep_module",
            output_dir=str(tmp_path),
            timeout_s=1,
            _test_behavior="sleep",
        )
        assert res.status == "timeout"
        assert res.timed_out is True
        assert "TimeoutError" in (res.error or "")

    # Verify events.jsonl
    assert events_file.exists()
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    assert any(e["module"] == "sleep_module" and e["status"] == "timeout" for e in events)


def test_traceback_recorded(tmp_path):
    """Ensure that an unhandled Python exception in a worker has its traceback written to events.jsonl."""
    events_file = tmp_path / "events.jsonl"

    with patch("neuralops.benchmarks.executor.discover_modules", return_value={
        "fail_module": MagicMock(type="attention", family="dense", supports_causal=True, supports_gqa=True, supports_rope=True, requires_cuda=False, timeout_s=5),
    }):
        res = execute_single(
            module="fail_module",
            output_dir=str(tmp_path),
            timeout_s=5,
            _test_behavior="fail",
        )
        assert res.status == "failed"
        assert "Deliberate internal kernel test error" in (res.error or "")

    assert events_file.exists()
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    fail_events = [e for e in events if e["module"] == "fail_module"]
    assert len(fail_events) > 0
    assert "Deliberate internal kernel test error" in fail_events[0]["traceback"]
