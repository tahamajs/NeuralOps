#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Dashboard router for NeuralOps Web UI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import get_doctor_status, get_recent_runs

router = APIRouter()


@router.get("/", response_class=HTMLResponse)
async def dashboard_page(request: Request) -> HTMLResponse:
    """Render main dashboard with telemetry overview and recent runs."""
    templates = request.app.state.templates
    status = get_doctor_status()
    recent_runs = get_recent_runs(limit=10)

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "active_tab": "dashboard",
            "status": status,
            "recent_runs": recent_runs,
        },
    )
