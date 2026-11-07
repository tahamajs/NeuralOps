#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Smoke tests verifying that the installed CLI entry point connects to create_app."""

import importlib
from starlette.testclient import TestClient

from neuralops.ui import create_app, cmd_ui, build_ui_parser
from neuralops.benchmarks.cli import build_parser, _dispatch


def test_installed_entrypoint_parser_has_web_ui_options():
    """Verify that build_parser configures the Web UI with all expected flags."""
    parser = build_parser()
    args = parser.parse_args(["ui", "--port", "8888", "--host", "127.0.0.1", "--no-browser"])
    assert args.subcommand == "ui"
    assert args.port == 8888
    assert args.host == "127.0.0.1"
    assert args.no_browser is True


def test_web_ui_create_app_initializes_routes():
    """Verify create_app instantiates FastAPI with valid application routes."""
    app = create_app()
    route_paths = []
    for r in app.routes:
        if hasattr(r, "path"):
            route_paths.append(r.path)
        elif hasattr(r, "original_router"):
            route_paths.extend([sub.path for sub in r.original_router.routes if hasattr(sub, "path")])

    # Check key registered endpoints
    assert "/" in route_paths
    assert "/attentions" in route_paths
    assert "/compressors" in route_paths
    assert "/doctor" in route_paths
    assert "/benchmarks" in route_paths


def test_web_ui_test_client_smoke():
    """Verify TestClient can execute a request against create_app without running a live server."""
    app = create_app()
    client = TestClient(app)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "NeuralOps" in resp.text
