#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""API routes for Observable Plot chart datasets."""

from __future__ import annotations

import json
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from neuralops.ui.queries.ledger import _get_connection

router = APIRouter(prefix="/api/chart")


@router.get("/roofline")
def get_roofline_data() -> JSONResponse:
    """Return arithmetic intensity vs latency / speedup data points for Roofline chart.

    Intensity is derived as (batch_size * seq_len * dim) / 1024.0 (proxy intensity).
    """
    query = """
        SELECT kernel_name, backend, batch_size, seq_len, dim, kernel_ms, speedup
        FROM hardware_kernel_shape_matrix
        WHERE status = 'PASS'
        ORDER BY kernel_name ASC, batch_size ASC, seq_len ASC
    """
    points: list[dict[str, Any]] = []
    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        for r in rows:
            b_sz = r["batch_size"] or 1
            s_len = r["seq_len"] or 128
            dim = r["dim"] or 64
            # Proxy intensity: operational volume normalized
            intensity = round((b_sz * s_len * dim) / 1024.0, 2)
            latency_ms = round(float(r["kernel_ms"]), 4) if r["kernel_ms"] is not None else 0.0
            speedup = round(float(r["speedup"]), 2) if r["speedup"] is not None else 1.0

            points.append(
                {
                    "module": r["kernel_name"],
                    "intensity": intensity,
                    "latency_ms": latency_ms,
                    "speedup": speedup,
                    "backend": r["backend"] or "unknown",
                }
            )

    return JSONResponse(content=points)


@router.get("/pareto")
def get_pareto_data() -> JSONResponse:
    """Return compression ratio vs accuracy and perplexity for Pareto frontier chart."""
    query = """
        SELECT module_name, accuracy, ppl, raw_json
        FROM results
        WHERE raw_json IS NOT NULL AND (accuracy IS NOT NULL OR ppl IS NOT NULL)
    """
    points: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()

    with _get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        for r in rows:
            module = r["module_name"]
            acc = float(r["accuracy"]) if r["accuracy"] is not None else None
            ppl_val = float(r["ppl"]) if r["ppl"] is not None else None

            # Parse compression_ratio from raw_json
            cr = 1.0
            if r["raw_json"]:
                try:
                    payload = json.loads(r["raw_json"])
                    metrics = payload.get("metrics", {})
                    if "compression_ratio" in metrics:
                        cr = round(float(metrics["compression_ratio"]), 4)
                except Exception:
                    pass

            # Filter duplicates for clean plotting
            key = (module, cr)
            if key in seen:
                continue
            seen.add(key)

            points.append(
                {
                    "module": module,
                    "compression_ratio": cr,
                    "accuracy": acc if acc is not None else 0.0,
                    "ppl": ppl_val if ppl_val is not None else 0.0,
                }
            )

    return JSONResponse(content=points)
