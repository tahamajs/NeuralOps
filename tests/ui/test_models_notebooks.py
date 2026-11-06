#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for Phase 8 Model Zoo and Notebook pages."""

import asyncio

from starlette.requests import Request

from neuralops.ui.queries.models_notebooks import list_model_zoo, list_notebooks
from neuralops.ui.routes.models_notebooks import models_page, notebooks_page
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


def test_list_notebooks_returns_94():
    """Verify list_notebooks queries exactly 94 canonical demonstration notebooks."""
    notebooks = list_notebooks()
    assert isinstance(notebooks, list)
    assert len(notebooks) == 94
    first = notebooks[0]
    assert "filename" in first
    assert "title" in first
    assert "description" in first
    assert "size_kb" in first
    assert "path" in first


def test_list_model_zoo_returns_models():
    """Verify list_model_zoo queries backbones and evaluation metrics."""
    models = list_model_zoo()
    assert isinstance(models, list)
    assert len(models) > 0
    first = models[0]
    assert "backbone" in first
    assert "name" in first
    assert "dataset" in first
    assert "evaluation_count" in first


def test_models_page_route():
    """Verify /models renders Model Zoo catalog."""
    req = _make_dummy_request("/models")
    resp = _run(models_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Model Zoo &amp; Backbones" in content or "Model Zoo & Backbones" in content
    assert "Architecture Regimes" in content


def test_notebooks_page_route():
    """Verify /notebooks renders all 94 tutorials and supports filtering."""
    req = _make_dummy_request("/notebooks")
    resp = _run(notebooks_page(req))
    assert resp.status_code == 200
    content = resp.body.decode("utf-8")
    assert "Demonstration Notebooks" in content
    assert "94" in content

    # Search filter
    filtered_resp = _run(notebooks_page(req, query="quickstart"))
    filtered_content = filtered_resp.body.decode("utf-8")
    assert "quickstart" in filtered_content.lower()
