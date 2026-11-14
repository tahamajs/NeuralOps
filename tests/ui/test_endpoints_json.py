#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for JSON API endpoints and port conflict handling in Web UI (F-60)."""

from __future__ import annotations

import argparse
import socket
import warnings
from unittest.mock import MagicMock, patch

warnings.filterwarnings("ignore", message=".*httpx.*")

from starlette.testclient import TestClient

from neuralops.ui.cli import cmd_ui, find_available_port, is_port_in_use
from neuralops.ui.server import create_app


def test_health_endpoint_returns_json():
    """Verify /health returns 200 and valid JSON."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


def test_runs_endpoint_returns_json():
    """Verify /runs returns 200 and structured JSON list."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/runs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "runs" in data
    assert isinstance(data["runs"], list)
    assert "total" in data


def test_root_endpoint_negotiates_json():
    """Verify GET / with Accept: application/json returns valid JSON."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/", headers={"accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "doctor" in data


def test_benchmarks_endpoint_negotiates_json():
    """Verify GET /benchmarks with Accept: application/json returns valid JSON."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/benchmarks", headers={"accept": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "results" in data
    assert isinstance(data["results"], list)
    assert "total" in data


def test_root_endpoint_default_returns_html():
    """Verify GET / without JSON accept header returns standard HTML dashboard."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "text/html" in resp.headers.get("content-type", "")
    assert "NeuralOps" in resp.text


def test_port_conflict_detection_and_resolution():
    """Verify is_port_in_use detects busy ports and find_available_port locates open port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]

        assert is_port_in_use("127.0.0.1", busy_port) is True

        open_port = find_available_port("127.0.0.1", busy_port)
        assert open_port != busy_port
        assert is_port_in_use("127.0.0.1", open_port) is False


def test_cmd_ui_handles_port_conflict_gracefully():
    """Verify cmd_ui detects occupied port and runs with next available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        busy_port = sock.getsockname()[1]

        args = argparse.Namespace(
            port=busy_port,
            host="127.0.0.1",
            share=False,
            static=None,
            no_browser=True,
            auth=None,
        )

        with patch("uvicorn.run") as mock_uvicorn:
            exit_code = cmd_ui(args)
            assert exit_code == 0
            assert mock_uvicorn.called
            # The port passed to uvicorn must not be the busy port
            call_port = mock_uvicorn.call_args[1].get("port") or mock_uvicorn.call_args[0][2]
            assert call_port != busy_port
