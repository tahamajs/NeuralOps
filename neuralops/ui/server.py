#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""FastAPI web server factory for NeuralOps Web UI."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import neuralops
from neuralops.ui.routes import (
    api_router,
    attentions_router,
    benchmarks_router,
    compressors_router,
    dashboard_router,
    doctor_router,
    models_notebooks_router,
    partials_router,
)


def create_app(auth_credentials: tuple[str, str] | None = None) -> FastAPI:
    """Create and configure the NeuralOps FastAPI application.

    Args:
        auth_credentials: Optional (username, password) tuple for Basic Auth.

    Returns:
        Configured FastAPI instance with static files, templates, and routers.
    """
    app = FastAPI(
        title="NeuralOps Web UI",
        version=neuralops.__version__,
        docs_url=None,
        redoc_url=None,
    )

    # Optional Basic Auth middleware
    if auth_credentials:
        from neuralops.ui.auth import BasicAuthMiddleware
        user, pwd = auth_credentials
        app.add_middleware(BasicAuthMiddleware, username=user, password=pwd)

    # Restrict CORS to localhost
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://127.0.0.1:8080", "http://localhost:8080"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    package_root = Path(__file__).resolve().parent
    templates_dir = package_root / "templates"
    static_dir = package_root / "static"

    # Store templates in app state for route handlers
    templates = Jinja2Templates(directory=str(templates_dir))
    app.state.templates = templates

    # Mount static assets
    if static_dir.is_dir():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Core health check endpoint
    @app.get("/health")
    async def health_check() -> dict[str, Any]:
        """Health check endpoint returning system status and package version."""
        return {"status": "ok", "version": neuralops.__version__}

    # Mount page routers
    app.include_router(dashboard_router)
    app.include_router(attentions_router)
    app.include_router(compressors_router)
    app.include_router(benchmarks_router)
    app.include_router(models_notebooks_router)
    app.include_router(doctor_router)
    app.include_router(partials_router)
    app.include_router(api_router)

    return app
