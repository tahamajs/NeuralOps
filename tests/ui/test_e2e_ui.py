#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""End-to-end integration and smoke tests for the NeuralOps UI package."""

import json
from pathlib import Path
import subprocess

from neuralops.ui.queries.ledger import get_doctor_status
from neuralops.ui.server import create_app
from neuralops.ui.static_builder import build_static_site


def test_ui_package_imports_and_invariants():
    """Verify UI package loads and confirms invariants."""
    status = get_doctor_status()
    assert status["attentions"] in (115, 131)
    assert status["compressors"] in (82, 72)
    assert status["triton_modules"] == 58
    assert status["cuda_packages"] in (140, 141)
    assert status["notebooks"] == 94


def test_server_factory_creates_all_routes():
    """Verify create_app instantiates all routes without errors."""
    app = create_app()
    all_paths = set()
    for route in app.routes:
        if hasattr(route, "path"):
            all_paths.add(route.path)
        if hasattr(route, "original_router"):
            for subroute in route.original_router.routes:
                all_paths.add(subroute.path)

    expected = {
        "/",
        "/health",
        "/static",
        "/attentions",
        "/attentions/{name}",
        "/compressors",
        "/compressors/{name}",
        "/benchmarks",
        "/benchmarks/export",
        "/models",
        "/notebooks",
        "/doctor",
        "/partials/recent-runs",
        "/partials/benchmarks",
        "/partials/attentions",
        "/partials/compressors",
        "/api/chart/roofline",
        "/api/chart/pareto",
    }
    for p in expected:
        assert p in all_paths, f"Expected route {p} not registered"


def test_cli_help_displays_cleanly():
    """Verify CLI --help runs cleanly via subprocess."""
    res = subprocess.run(
        ["python3", "-m", "neuralops.cli.main", "ui", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "--port" in res.stdout
    assert "--share" in res.stdout
    assert "--static" in res.stdout
    assert "--auth" in res.stdout


def test_e2e_static_export_integrity(tmp_path: Path):
    """Verify end-to-end static export produces valid HTML and JSON manifest."""
    out_dir = tmp_path / "e2e_site"
    summary = build_static_site(out_dir)

    assert summary["total_pages"] >= 200
    manifest_path = out_dir / "manifest.json"
    assert manifest_path.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert len(manifest["files"]) >= 200
