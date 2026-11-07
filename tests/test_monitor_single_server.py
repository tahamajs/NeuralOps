#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests verifying single monitor / Web UI server unification (F-59).
"""

from __future__ import annotations

from fastapi import FastAPI

from neuralops.runners.monitor.fastapi_app import create_app as monitor_create_app
from neuralops.ui.server import create_app as ui_create_app


def test_monitor_and_ui_are_same_app() -> None:
    """Verify runners/monitor/fastapi_app.py delegates directly to canonical ui/server.py."""
    app_monitor = monitor_create_app()
    app_ui = ui_create_app()

    assert isinstance(app_monitor, FastAPI)
    assert isinstance(app_ui, FastAPI)
    assert app_monitor.title == app_ui.title
    assert app_monitor.version == app_ui.version
    assert len(app_monitor.routes) == len(app_ui.routes)

    # Route paths match between monitor app and canonical Web UI
    monitor_paths = [r.path for r in app_monitor.routes if hasattr(r, "path")]
    ui_paths = [r.path for r in app_ui.routes if hasattr(r, "path")]
    assert monitor_paths == ui_paths
