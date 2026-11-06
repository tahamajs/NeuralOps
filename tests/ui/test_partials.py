#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 6 HTMX partial rendering endpoints."""

import asyncio

from starlette.requests import Request

from neuralops.ui.routes.partials import (
    partial_attentions,
    partial_benchmarks,
    partial_compressors,
    partial_recent_runs,
)
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


def test_partial_recent_runs():
    """Verify partial_recent_runs returns rendered HTML table rows."""
    req = _make_dummy_request("/partials/recent-runs")
    resp = _run(partial_recent_runs(req, limit=5))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "<tr>" in content
    assert "<td>" in content


def test_partial_benchmarks():
    """Verify partial_benchmarks returns table and pagination controls."""
    req = _make_dummy_request("/partials/benchmarks")
    resp = _run(partial_benchmarks(req, page=1, page_size=20))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "data-table" in content
    assert "hx-target=\"#benchmarks-table-container\"" in content
    assert "Page 1 of" in content


def test_partial_attentions_search_and_filter():
    """Verify partial_attentions filters correctly on query and family."""
    req = _make_dummy_request("/partials/attentions")
    # All attentions
    resp = _run(partial_attentions(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "cosformer" in content

    # Search query
    resp_query = _run(partial_attentions(req, query="cosformer"))
    content_query = resp_query.body.decode("utf-8")
    assert "cosformer" in content_query
    assert "fourier_attention" not in content_query


def test_partial_compressors_search_and_filter():
    """Verify partial_compressors filters correctly on query and paradigm."""
    req = _make_dummy_request("/partials/compressors")
    # All compressors
    resp = _run(partial_compressors(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "direct_svd" in content

    # Search query
    resp_query = _run(partial_compressors(req, query="direct_svd"))
    content_query = resp_query.body.decode("utf-8")
    assert "direct_svd" in content_query
    assert "flat_llm" not in content_query
