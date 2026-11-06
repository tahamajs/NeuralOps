#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Web UI FastAPI Server Module.

Provides minimal, production-grade web dashboard and interactive exploration interface.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import neuralops


def create_app() -> FastAPI:
    """Construct and configure the minimal NeuralOps Web UI FastAPI application."""
    base_dir = Path(__file__).resolve().parent
    templates_dir = base_dir / "templates"
    static_dir = base_dir / "static"

    app = FastAPI(
        title="NeuralOps Web UI",
        description="Publication-grade dashboard and interactive explorer for NeuralOps.",
        version=neuralops.__version__,
        docs_url="/docs",
        redoc_url=None,
    )

    # CORS middleware restricted to localhost / loopback interfaces
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost",
            "http://127.0.0.1",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ],
        allow_origin_regex=r"^http://(localhost|127\.0\.0\.1)(:[0-9]+)?$",
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # Static files mount
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # Jinja2 templates configuration
    templates = Jinja2Templates(directory=str(templates_dir))

    @app.get("/health", response_class=JSONResponse, tags=["Diagnostics"])
    async def health_check() -> Dict[str, Any]:
        """Health check endpoint returning system status and package version."""
        return {
            "status": "ok",
            "version": neuralops.__version__,
        }

    @app.get("/", response_class=HTMLResponse, tags=["Dashboard"])
    async def root_view(request: Request) -> HTMLResponse:
        """Root view rendering placeholder template."""
        return templates.TemplateResponse(
            request=request,
            name="placeholder.html",
        )

    return app


__all__ = ["create_app"]
