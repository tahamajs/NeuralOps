#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""HTMX partial rendering endpoints for dynamic live UI updates."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import get_recent_runs, get_results
from neuralops.ui.queries.static_catalog import list_attentions, list_compressors

router = APIRouter(prefix="/partials")


@router.get("/recent-runs", response_class=HTMLResponse)
async def partial_recent_runs(request: Request, limit: int = 10) -> HTMLResponse:
    """Return rendered table rows for the recent benchmark executions."""
    templates = request.app.state.templates
    recent_runs = get_recent_runs(limit=limit)
    return templates.TemplateResponse(
        request=request,
        name="partials/recent_runs_rows.html",
        context={"recent_runs": recent_runs},
    )


@router.get("/benchmarks", response_class=HTMLResponse)
async def partial_benchmarks(
    request: Request,
    module_name: str | None = None,
    status: str | None = None,
    family: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> HTMLResponse:
    """Return rendered benchmark table fragment with pagination."""
    templates = request.app.state.templates
    offset = max(0, (page - 1) * page_size)

    filters: dict[str, Any] = {}
    if module_name:
        filters["module_name"] = module_name.strip()
    if status:
        filters["status"] = status.strip()
    if family:
        filters["family"] = family.strip()

    rows, total_count = get_results(filters=filters, limit=page_size, offset=offset)
    total_pages = max(1, (total_count + page_size - 1) // page_size)

    return templates.TemplateResponse(
        request=request,
        name="partials/benchmarks_table.html",
        context={
            "rows": rows,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "filter_module_name": module_name or "",
            "filter_status": status or "",
            "filter_family": family or "",
        },
    )


@router.get("/attentions", response_class=HTMLResponse)
async def partial_attentions(
    request: Request,
    family: str | None = None,
    query: str | None = None,
) -> HTMLResponse:
    """Return rendered attentions table fragment filtered by family or search query."""
    templates = request.app.state.templates
    attentions = list_attentions(family=family if family else None)
    if query:
        q = query.strip().lower()
        attentions = [
            a for a in attentions
            if q in a["name"].lower() or q in a["doc"].lower()
        ]

    return templates.TemplateResponse(
        request=request,
        name="partials/attentions_table.html",
        context={"attentions": attentions},
    )


@router.get("/compressors", response_class=HTMLResponse)
async def partial_compressors(
    request: Request,
    paradigm: str | None = None,
    query: str | None = None,
) -> HTMLResponse:
    """Return rendered compressors table fragment filtered by paradigm or search query."""
    templates = request.app.state.templates
    compressors = list_compressors(paradigm=paradigm if paradigm else None)
    if query:
        q = query.strip().lower()
        compressors = [
            c for c in compressors
            if q in c["name"].lower() or q in c["doc"].lower()
        ]

    return templates.TemplateResponse(
        request=request,
        name="partials/compressors_table.html",
        context={"compressors": compressors},
    )
