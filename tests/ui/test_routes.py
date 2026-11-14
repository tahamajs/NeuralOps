#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 5 UI routes."""

import asyncio

from starlette.requests import Request

from neuralops.ui.routes.attentions import attention_detail_page, list_attentions_page
from neuralops.ui.routes.benchmarks import benchmarks_page, export_benchmarks_csv
from neuralops.ui.routes.compressors import compressor_detail_page, list_compressors_page
from neuralops.ui.routes.dashboard import dashboard_page
from neuralops.ui.routes.doctor import doctor_page
from neuralops.ui.server import create_app


def _run(coro):
    """Run an async function synchronously."""
    return asyncio.run(coro)


def _make_dummy_request(path: str = "/") -> Request:
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


def test_dashboard_route():
    """Verify dashboard route renders telemetry overview and recent runs."""
    req = _make_dummy_request("/")
    resp = _run(dashboard_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Operational Overview" in content
    assert "Attention Registry" in content
    assert "Recent Benchmark Executions" in content


def test_attentions_list_route():
    """Verify attentions catalog route lists attention mechanisms."""
    req = _make_dummy_request("/attentions")
    resp = _run(list_attentions_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Attention Mechanism Registry" in content
    assert "cosformer" in content
    assert "linear_attention" in content


def test_attentions_filter_route():
    """Verify filtering attentions by family."""
    req = _make_dummy_request("/attentions?family=linear")
    resp = _run(list_attentions_page(req, family="linear"))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "linear" in content


def test_attention_detail_route():
    """Verify attention detail route renders metadata and execution history."""
    req = _make_dummy_request("/attentions/cosformer")
    resp = _run(attention_detail_page(req, name="cosformer"))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "cosformer" in content
    assert "Module:" in content
    assert "Execution History" in content


def test_compressors_list_route():
    """Verify compressors catalog route lists compressors."""
    req = _make_dummy_request("/compressors")
    resp = _run(list_compressors_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Model Compressor Registry" in content
    assert "direct_svd" in content


def test_compressor_detail_route():
    """Verify compressor detail route renders metadata and execution history."""
    req = _make_dummy_request("/compressors/direct_svd")
    resp = _run(compressor_detail_page(req, name="direct_svd"))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "direct_svd" in content
    assert "Execution History" in content


def test_benchmarks_route_and_export():
    """Verify benchmarks table pagination and CSV export."""
    req = _make_dummy_request("/benchmarks")
    resp = _run(benchmarks_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Benchmark Ledger Explorer" in content
    assert "Export CSV" in content

    # Test CSV export
    csv_resp = _run(export_benchmarks_csv(status="active"))
    assert csv_resp.status_code == 200
    assert csv_resp.media_type == "text/csv"
    csv_content = csv_resp.body.decode("utf-8")
    assert "module_name" in csv_content


def test_doctor_route():
    """Verify doctor route displays system health and invariants."""
    req = _make_dummy_request("/doctor")
    resp = _run(doctor_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "System Doctor &amp; Diagnostics" in content or "System Doctor & Diagnostics" in content
    assert "ALL INVARIANTS PASS" in content
    assert "115" in content or "131" in content
    assert "82" in content or "72" in content
    assert "58" in content
    assert "140" in content or "141" in content
    assert "94" in content
