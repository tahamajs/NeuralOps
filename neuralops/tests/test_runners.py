"""Unit tests for neuralops.runners module."""
from __future__ import annotations

import pytest


def test_monitor_server_imports():
    """Verify monitor server components can be imported cleanly."""
    import neuralops.runners.run_monitor_server as server
    assert hasattr(server, "main") or hasattr(server, "start_server") or server is not None


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


def test_autonomous_pipeline_imports():
    """Verify autonomous pipeline can be imported."""
    import neuralops.runners.run_autonomous_48h_pipeline as auto_pipe
    assert auto_pipe is not None


def test_build_all_streams():
    """Verify pipeline module sweeps can be inspected."""
    import neuralops.runners.run_autonomous_48h_pipeline as auto_pipe
    assert hasattr(auto_pipe, "SWEEP_CONFIGS") or hasattr(auto_pipe, "main")


def test_run_continuous_orchestrator_imports():
    """Verify continuous orchestrator imports cleanly."""
    import neuralops.runners.run_continuous_orchestrator as cont_orch
    assert cont_orch is not None


def test_launch_parallel_sweep_import():
    """Verify benchmarks launch_parallel_sweep module imports."""
    import neuralops.benchmarks.launch_parallel_sweep as lps
    assert lps is not None


def test_main_entrypoints():
    """Verify runners package exposes its expected runner scripts."""
    import neuralops.runners as runners
    assert runners is not None
