#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""Model Zoo and Notebook routes for NeuralOps Web UI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.models_notebooks import list_model_zoo, list_notebooks

router = APIRouter()


@router.get("/models", response_class=HTMLResponse)
async def models_page(request: Request) -> HTMLResponse:
    """Render Model Zoo catalog and evaluation benchmarks."""
    templates = request.app.state.templates
    models = list_model_zoo()

    return templates.TemplateResponse(
        request=request,
        name="models.html",
        context={
            "active_tab": "models",
            "models": models,
            "total_count": len(models),
        },
    )


@router.get("/notebooks", response_class=HTMLResponse)
async def notebooks_page(request: Request, query: str | None = None) -> HTMLResponse:
    """Render canonical demonstration tutorials (94 notebooks)."""
    templates = request.app.state.templates
    all_notebooks = list_notebooks()

    if query:
        q = query.strip().lower()
        notebooks = [
            nb for nb in all_notebooks
            if q in nb["title"].lower() or q in nb["description"].lower() or q in nb["filename"].lower()
        ]
    else:
        notebooks = all_notebooks

    return templates.TemplateResponse(
        request=request,
        name="notebooks.html",
        context={
            "active_tab": "notebooks",
            "notebooks": notebooks,
            "total_count": len(all_notebooks),
            "search_query": query or "",
        },
    )
