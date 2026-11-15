#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""FastAPI web server factory for NeuralOps Web UI."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
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

logger = logging.getLogger("NeuralOpsWebUI")


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

    # Content negotiation middleware for JSON API callers
    @app.middleware("http")
    async def content_negotiation_middleware(request: Request, call_next: Any) -> Response:
        accept = request.headers.get("accept", "")
        format_param = request.query_params.get("format", "")
        path = request.url.path

        if "application/json" in accept or format_param == "json":
            if path in ("/", ""):
                from neuralops.ui.queries.ledger import get_doctor_status, get_recent_runs

                try:
                    status = get_doctor_status()
                except Exception as e:
                    logger.debug("Doctor query fallback: %s", e)
                    status = {"version": neuralops.__version__}
                try:
                    recent_runs = get_recent_runs(limit=10)
                except Exception as e:
                    logger.debug("Recent runs query fallback: %s", e)
                    recent_runs = []
                return JSONResponse(
                    {
                        "status": "ok",
                        "version": neuralops.__version__,
                        "doctor": status,
                        "recent_runs": recent_runs,
                    }
                )
            if path == "/benchmarks":
                from neuralops.ui.queries.ledger import get_results

                try:
                    page = int(request.query_params.get("page", 1))
                    page_size = int(request.query_params.get("page_size", 50))
                    offset = max(0, (page - 1) * page_size)
                    filters = {}
                    if request.query_params.get("module_name"):
                        filters["module_name"] = request.query_params.get("module_name", "").strip()
                    if request.query_params.get("status"):
                        filters["status"] = request.query_params.get("status", "").strip()
                    if request.query_params.get("family"):
                        filters["family"] = request.query_params.get("family", "").strip()
                    rows, total = get_results(filters=filters, limit=page_size, offset=offset)
                except Exception as e:
                    logger.debug("Benchmarks query fallback: %s", e)
                    rows, total, page, page_size = [], 0, 1, 50
                return JSONResponse(
                    {
                        "status": "ok",
                        "total": total,
                        "page": page,
                        "page_size": page_size,
                        "results": rows,
                    }
                )

        return await call_next(request)

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

    # Core benchmark runs endpoint
    @app.get("/runs")
    async def runs_endpoint(limit: int = 50) -> dict[str, Any]:
        """Runs endpoint returning recent benchmark runs as JSON."""
        from neuralops.ui.queries.ledger import get_recent_runs

        try:
            runs = get_recent_runs(limit=limit)
        except Exception as e:
            logger.debug("Failed to query recent runs: %s", e)
            runs = []
        return {"status": "ok", "total": len(runs), "runs": runs}

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
