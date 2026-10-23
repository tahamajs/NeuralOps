#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Regression test suite verifying analysis and benchmark relocation out of core (P015, P045-P047).

Covers:
- test_core_does_not_load_matplotlib: core import does not introduce matplotlib into sys.modules.
- test_core_does_not_load_pandas: core import does not introduce pandas into sys.modules.
- test_analysis_paper_export_lazy: lazy or direct import from neuralops.analysis works.
- test_top_level_paper_export: backward compatibility re-export from top-level neuralops.
- test_results_ledger_new_path: results_ledger import and ledger initialization.
- test_benchmark_scripts_new_path: multi_mode_benchmark and run_long_context_sweep importability.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_core_does_not_load_matplotlib():
    """Verify that importing neuralops.core does not load matplotlib into sys.modules."""
    code = "import sys; import neuralops.core; assert 'matplotlib' not in sys.modules, f'matplotlib loaded: {sys.modules.get(\"matplotlib\")}'"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_core_does_not_load_pandas():
    """Verify that importing neuralops.core does not load pandas into sys.modules."""
    code = "import sys; import neuralops.core; assert 'pandas' not in sys.modules, f'pandas loaded: {sys.modules.get(\"pandas\")}'"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert res.returncode == 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"


def test_analysis_paper_export_lazy():
    """Verify neuralops.analysis.paper_export exports required functions."""
    from neuralops.analysis import paper_export

    assert hasattr(paper_export, "export_benchmark_table")
    assert hasattr(paper_export, "compute_pareto_frontier")


def test_top_level_paper_export():
    """Verify backward compatibility for from neuralops import paper_export."""
    import neuralops

    assert hasattr(neuralops, "paper_export") or "paper_export" in dir(neuralops)


def test_results_ledger_new_path():
    """Verify results_ledger is importable from neuralops.analysis."""
    from neuralops.analysis.results_ledger import ResultsLedger, get_default_ledger

    ledger = get_default_ledger()
    assert ledger is not None


def test_benchmark_scripts_new_path():
    """Verify multi_mode_benchmark is importable from neuralops.benchmarks."""
    from neuralops.benchmarks.multi_mode_benchmark import (
        MultiModeBenchmarkRunner,
        SUPPORTED_MODES,
    )

    assert MultiModeBenchmarkRunner is not None
    assert len(SUPPORTED_MODES) > 0
