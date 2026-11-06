#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Benchmarks explorer and export router for NeuralOps Web UI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request, Response
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import export_results_csv, get_results
from neuralops.ui.queries.static_catalog import list_families

router = APIRouter()


@router.get("/benchmarks", response_class=HTMLResponse)
async def benchmarks_page(
    request: Request,
    module_name: str | None = None,
    status: str | None = None,
    family: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> HTMLResponse:
    """Render benchmark results table with filtering and pagination."""
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
        name="benchmarks.html",
        context={
            "active_tab": "benchmarks",
            "rows": rows,
            "total_count": total_count,
            "page": page,
            "page_size": page_size,
            "total_pages": total_pages,
            "families": list_families(),
            "filter_module_name": module_name or "",
            "filter_status": status or "",
            "filter_family": family or "",
        },
    )


@router.get("/benchmarks/export", response_class=Response)
async def export_benchmarks_csv(
    module_name: str | None = None,
    status: str | None = None,
    family: str | None = None,
) -> Response:
    """Export benchmark rows matching filters as a downloadable CSV."""
    filters: dict[str, Any] = {}
    if module_name:
        filters["module_name"] = module_name.strip()
    if status:
        filters["status"] = status.strip()
    if family:
        filters["family"] = family.strip()

    csv_data = export_results_csv(filters=filters)
    return Response(
        content=csv_data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=neuralops_benchmark_results.csv"},
    )
