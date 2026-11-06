#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Unit tests for UI server templates and static assets without httpx dependency warnings."""

from pathlib import Path

from fastapi.templating import Jinja2Templates

from neuralops.ui.server import create_app


def test_static_css_file_exists_and_valid():
    """Verify ui.css exists and contains primary styles."""
    repo_root = Path(__file__).resolve().parents[2]
    css_path = repo_root / "neuralops" / "ui" / "static" / "ui.css"
    assert css_path.is_file()
    content = css_path.read_text(encoding="utf-8")
    assert "--bg-primary" in content
    assert "app-header" in content


def test_static_htmx_file_exists_and_valid():
    """Verify local vendored htmx.min.js exists and is complete (>40KB)."""
    repo_root = Path(__file__).resolve().parents[2]
    htmx_path = repo_root / "neuralops" / "ui" / "static" / "htmx.min.js"
    assert htmx_path.is_file()
    assert htmx_path.stat().st_size > 40000


def test_base_template_structure():
    """Verify base.html renders with navigation and links."""
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "neuralops" / "ui" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    template = templates.get_template("base.html")
    rendered = template.render(active_tab="dashboard", version="2.0.6")

    assert "NeuralOps" in rendered
    assert "/static/ui.css" in rendered
    assert "/static/htmx.min.js" in rendered
    assert "Dashboard" in rendered
    assert "Attentions" in rendered
    assert "Compressors" in rendered
    assert "Benchmarks" in rendered
    assert "Doctor" in rendered


def test_placeholder_template_extends_base():
    """Verify placeholder.html extends base and renders dashboard card."""
    repo_root = Path(__file__).resolve().parents[2]
    templates_dir = repo_root / "neuralops" / "ui" / "templates"
    templates = Jinja2Templates(directory=str(templates_dir))
    template = templates.get_template("placeholder.html")
    rendered = template.render(active_tab="dashboard", version="2.0.6")

    assert "System Status" in rendered
    assert "131" in rendered
    assert "72" in rendered
    assert "58" in rendered
    assert "141" in rendered
    assert "/static/ui.css" in rendered


def test_server_static_and_template_configuration():
    """Verify create_app configures static routes and Jinja2 templates."""
    app = create_app()
    routes = [route.path for route in app.routes]
    assert "/static" in routes
    assert "/" in routes
    assert "/health" in routes
