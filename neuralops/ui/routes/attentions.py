#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Attentions catalog router for NeuralOps Web UI."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import get_attention_results
from neuralops.ui.queries.static_catalog import (
    get_attention_metadata,
    list_attentions,
    list_families,
)

router = APIRouter()


@router.get("/attentions", response_class=HTMLResponse)
async def list_attentions_page(request: Request, family: str | None = None) -> HTMLResponse:
    """Render catalog table of attention mechanisms with optional family filter."""
    templates = request.app.state.templates
    families = list_families()
    attentions = list_attentions(family=family if family else None)

    return templates.TemplateResponse(
        request=request,
        name="attentions.html",
        context={
            "active_tab": "attentions",
            "attentions": attentions,
            "families": families,
            "selected_family": family or "",
            "total_count": len(attentions),
        },
    )


@router.get("/attentions/{name}", response_class=HTMLResponse)
async def attention_detail_page(request: Request, name: str) -> HTMLResponse:
    """Render detail page for a specific attention mechanism."""
    templates = request.app.state.templates
    metadata = get_attention_metadata(name)
    if not metadata:
        raise HTTPException(status_code=404, detail=f"Attention '{name}' not found")

    benchmark_history = get_attention_results(name)

    return templates.TemplateResponse(
        request=request,
        name="attention_detail.html",
        context={
            "active_tab": "attentions",
            "attention": metadata,
            "history": benchmark_history,
        },
    )
