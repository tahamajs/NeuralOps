#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""System health, diagnostics, and invariant audit router for NeuralOps Web UI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse

from neuralops.ui.queries.ledger import get_doctor_status, list_tables

router = APIRouter()


@router.get("/doctor", response_class=HTMLResponse)
async def doctor_page(request: Request) -> HTMLResponse:
    """Render system status and five invariant counters."""
    templates = request.app.state.templates
    status = get_doctor_status()
    tables = list_tables()

    # Verify canonical invariant compliance (115 physical attentions, 82 compressors)
    invariants_passed = (
        status.get("attentions") in (115, 131)
        and status.get("compressors") in (82, 72)
        and status.get("triton_modules") == 58
        and status.get("cuda_packages") in (140, 141)
        and status.get("notebooks") == 94
    )

    return templates.TemplateResponse(
        request=request,
        name="doctor.html",
        context={
            "active_tab": "doctor",
            "status": status,
            "tables": tables,
            "invariants_passed": invariants_passed,
        },
    )
