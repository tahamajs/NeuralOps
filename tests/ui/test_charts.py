#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 7 Observable Plot chart endpoints and visualizations."""

import asyncio
import json

from starlette.requests import Request

from neuralops.ui.routes.api import get_pareto_data, get_roofline_data
from neuralops.ui.routes.benchmarks import benchmarks_page
from neuralops.ui.server import create_app


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


def _make_dummy_request(path: str) -> Request:
    """Create a mock Request bound to the initialized FastAPI app."""
    app = create_app()
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": [],
        "app": app,
    }
    return Request(scope)


def test_roofline_api_returns_list():
    """Verify GET /api/chart/roofline returns 200 and a list with required keys."""
    resp = get_roofline_data()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "module" in first
    assert "intensity" in first
    assert "latency_ms" in first
    assert "speedup" in first
    assert "backend" in first


def test_pareto_api_returns_list():
    """Verify GET /api/chart/pareto returns 200 and a list with required keys."""
    resp = get_pareto_data()
    assert resp.status_code == 200
    data = json.loads(resp.body.decode("utf-8"))
    assert isinstance(data, list)
    assert len(data) > 0
    first = data[0]
    assert "module" in first
    assert "compression_ratio" in first
    assert "accuracy" in first
    assert "ppl" in first


def test_benchmarks_page_contains_chart_divs():
    """Verify GET /benchmarks HTML contains roofline-chart, pareto-chart, and charts.js script."""
    req = _make_dummy_request("/benchmarks")
    resp = _run(benchmarks_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert 'id="roofline-chart"' in content
    assert 'id="pareto-chart"' in content
    assert 'src="/static/charts.js"' in content
