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
