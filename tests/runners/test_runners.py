#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Unit tests for neuralops.runners module."""
from __future__ import annotations

import pytest


def test_monitor_stats_collection():
    """Verify monitor subpackage imports."""
    from neuralops.runners.monitor import (
        telemetry,
        stats,
    )
    assert telemetry is not None
    assert stats is not None


def test_unified_orchestrator_imports():
    """Verify unified orchestrator can be imported cleanly."""
    import neuralops.runners.unified_orchestrator as orchestrator
    assert orchestrator is not None


def test_main_entrypoints():
    """Verify runners package exposes its expected runner scripts."""
    import neuralops.runners as runners
    assert hasattr(runners, "UnifiedOrchestrator")


def test_run_monitor_server_cross_platform_paths(monkeypatch: pytest.MonkeyPatch):
    """Verify monitor server configuration is OS-agnostic and responds to env vars."""
    import importlib
    monkeypatch.setenv("IS_LOCAL_GPU", "true")
    import neuralops.runners.run_monitor_server as srv
    importlib.reload(srv)
    assert isinstance(srv.IS_LOCAL_GPU, bool)
    assert srv.IS_LOCAL_GPU is True

    monkeypatch.setenv("IS_LOCAL_GPU", "false")
    importlib.reload(srv)
    assert srv.IS_LOCAL_GPU is False


def test_run_master_pipeline_missing_script_raises_descriptive_error():
    """Verify run_task raises a descriptive FileNotFoundError before running missing scripts."""
    from neuralops.runners.run_master_pipeline import run_task

    dummy_task = {
        "task": "zeroshot",
        "module": "dummy_mod",
        "dataset": "sst2",
        "backbone": "gpt2",
    }
    with pytest.raises(FileNotFoundError, match="Target script for task 'zeroshot' does not exist"):
        run_task(dummy_task)


def test_unified_orchestrator_exit_code_validation(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Verify that unified orchestrator marks status as FAIL on non-zero exit code."""
    import subprocess
    from pathlib import Path
    from unittest.mock import MagicMock
    import neuralops.runners.unified_orchestrator as orch

    mock_proc = MagicMock()
    mock_proc.returncode = 1
    mock_proc.stderr = "CUDA out of memory"

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: mock_proc)
    monkeypatch.setattr(orch, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(orch, "MODELS", [{"name": "TestModel", "slug": "test_m", "model": "test", "family": "bert"}])

    # Ensure TRAINER_SCRIPT exists for test
    dummy_trainer = tmp_path / "trainer.py"
    dummy_trainer.write_text("print('test')")
    monkeypatch.setattr(orch, "TRAINER_SCRIPT", dummy_trainer)

    orch.run_orchestrator("dummy_module", default_args=[])
    summary_csv = tmp_path / "results" / "01_unified_runs" / "unified_dummy_module" / "4models_summary.csv"
    assert summary_csv.exists()
    import csv
    with summary_csv.open() as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 1
        assert rows[0]["status"] == "FAIL"
        assert rows[0]["model_name"] == "TestModel"
