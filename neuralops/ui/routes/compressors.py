#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Compressors catalog router for NeuralOps Web UI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import get_compressor_results
from neuralops.ui.queries.static_catalog import (
    get_compressor_metadata,
    list_compressors,
    list_paradigms,
)

router = APIRouter()


@router.get("/compressors", response_class=HTMLResponse)
async def list_compressors_page(request: Request, paradigm: str | None = None) -> HTMLResponse:
    """Render catalog table of compressors with optional paradigm filter."""
    templates = request.app.state.templates
    paradigms = list_paradigms()
    compressors = list_compressors(paradigm=paradigm if paradigm else None)

    return templates.TemplateResponse(
        request=request,
        name="compressors.html",
        context={
            "active_tab": "compressors",
            "compressors": compressors,
            "paradigms": paradigms,
            "selected_paradigm": paradigm or "",
            "total_count": len(compressors),
        },
    )


@router.get("/compressors/{name}", response_class=HTMLResponse)
async def compressor_detail_page(request: Request, name: str) -> HTMLResponse:
    """Render detail page for a specific compressor."""
    templates = request.app.state.templates
    metadata = get_compressor_metadata(name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Compressor '{name}' not found")

    benchmark_history = get_compressor_results(name)

    return templates.TemplateResponse(
        request=request,
        name="compressor_detail.html",
        context={
            "active_tab": "compressors",
            "compressor": metadata,
            "history": benchmark_history,
        },
    )
