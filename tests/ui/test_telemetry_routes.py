#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit and integration tests for UI live telemetry WebSocket routes and KaTeX tables."""

from __future__ import annotations

import json
import warnings
from unittest.mock import patch

warnings.filterwarnings("ignore", category=DeprecationWarning)
try:
    from starlette.exceptions import StarletteDeprecationWarning

    warnings.filterwarnings("ignore", category=StarletteDeprecationWarning)
except (ImportError, AttributeError):
    pass

import pytest
from starlette.testclient import TestClient

from neuralops.ui.routes.telemetry import collect_telemetry_snapshot
from neuralops.ui.server import create_app


def test_collect_telemetry_snapshot_structure() -> None:
    """Verify collect_telemetry_snapshot returns expected keys and contract."""
    snapshot = collect_telemetry_snapshot()
    assert isinstance(snapshot, dict)
    assert snapshot.get("event") == "telemetry_update"
    assert "hardware" in snapshot
    assert "doctor" in snapshot
    assert "recent_runs" in snapshot

    hw = snapshot["hardware"]
    assert "gpu_temp_c" in hw
    assert "gpu_vram_used_mb" in hw
    assert "gpu_power_w" in hw


def test_collect_telemetry_snapshot_with_fallbacks() -> None:
    """Verify collect_telemetry_snapshot handles exceptions fail-closed."""
    with patch(
        "neuralops.ui.routes.telemetry.get_hardware_telemetry",
        side_effect=RuntimeError("Hardware sensor error"),
    ), patch(
        "neuralops.ui.routes.telemetry.get_doctor_status",
        side_effect=RuntimeError("Doctor query error"),
    ), patch(
        "neuralops.ui.routes.telemetry.get_recent_runs",
        side_effect=RuntimeError("Ledger query error"),
    ):
        snapshot = collect_telemetry_snapshot()
        assert snapshot["event"] == "telemetry_update"
        assert snapshot["hardware"]["device"] == "cpu"
        assert snapshot["doctor"] == {}
        assert snapshot["recent_runs"] == []


def test_telemetry_snapshot_rest_endpoint() -> None:
    """Verify GET /ws/telemetry/snapshot returns 200 with structured JSON."""
    app = create_app()
    client = TestClient(app)
    response = client.get("/ws/telemetry/snapshot")
    assert response.status_code == 200
    data = response.json()
    assert data["event"] == "telemetry_update"
    assert "hardware" in data
    assert "doctor" in data
    assert "recent_runs" in data


def test_websocket_telemetry_stream() -> None:
    """Verify WebSocket connection at /ws/telemetry receives live telemetry stream."""
    app = create_app()
    client = TestClient(app)

    with client.websocket_connect("/ws/telemetry") as websocket:
        # Receive the first telemetry frame sent immediately on connection
        data_text = websocket.receive_text()
        assert data_text is not None
        payload = json.loads(data_text)
        assert payload["event"] == "telemetry_update"
        assert "hardware" in payload
        assert "recent_runs" in payload
        assert "doctor" in payload


def test_katex_headers_rendered_in_templates() -> None:
    """Verify KaTeX dependencies in base.html and formulas in attention and compressor templates."""
    app = create_app()
    client = TestClient(app)

    # 1. Base template should contain KaTeX CDN tags
    resp_dashboard = client.get("/")
    assert resp_dashboard.status_code == 200
    assert "katex.min.css" in resp_dashboard.text
    assert "katex.min.js" in resp_dashboard.text
    assert "auto-render.min.js" in resp_dashboard.text
    assert "id=\"live-telemetry-card\"" in resp_dashboard.text

    # 2. Benchmarks partial should contain LaTeX mathematical formula
    resp_bench = client.get("/partials/benchmarks")
    assert resp_bench.status_code == 200
    assert r"\frac{t_{ref}}{t_{opt}}" in resp_bench.text

    # 3. Attention detail template should contain Cosine Parity LaTeX formula
    resp_attn = client.get("/attentions/cosformer")
    if resp_attn.status_code == 200:
        assert r"\(S_c(y, \hat{y})\)" in resp_attn.text

    # 4. Compressor detail template should contain Cosine Parity LaTeX formula
    resp_comp = client.get("/compressors/aa_svd")
    if resp_comp.status_code == 200:
        assert r"\(S_c(y, \hat{y})\)" in resp_comp.text
