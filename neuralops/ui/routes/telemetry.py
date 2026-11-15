#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors

"""WebSocket and real-time streaming telemetry routes for NeuralOps Web UI."""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from neuralops.benchmarks.telemetry.hardware import get_hardware_telemetry
from neuralops.ui.queries.ledger import get_doctor_status, get_recent_runs

logger = logging.getLogger("NeuralOpsWebUI.Telemetry")

router = APIRouter(prefix="/ws")


def collect_telemetry_snapshot() -> dict[str, Any]:
    """Capture a complete snapshot of hardware telemetry and system health.

    Returns:
        Structured dictionary containing hardware metrics, doctor status,
        and latest benchmark execution runs.
    """
    try:
        hw_telemetry = get_hardware_telemetry()
    except Exception as exc:
        logger.debug("Hardware telemetry capture fallback: %s", exc)
        hw_telemetry = {
            "gpu_temp_c": 0.0,
            "gpu_vram_used_mb": 0.0,
            "gpu_power_w": 0.0,
            "cpu_util_pct": 0.0,
            "device": "cpu",
        }

    try:
        doctor_status = get_doctor_status()
    except Exception as exc:
        logger.debug("Doctor status capture fallback: %s", exc)
        doctor_status = {}

    try:
        recent_runs = get_recent_runs(limit=5)
    except Exception as exc:
        logger.debug("Recent runs capture fallback: %s", exc)
        recent_runs = []

    return {
        "event": "telemetry_update",
        "hardware": hw_telemetry,
        "doctor": doctor_status,
        "recent_runs": recent_runs,
    }


@router.websocket("/telemetry")
async def websocket_telemetry_endpoint(websocket: WebSocket) -> None:
    """Stream real-time hardware telemetry and benchmark ledger updates via WebSocket.

    Args:
        websocket: FastAPI WebSocket connection instance.
    """
    await websocket.accept()
    logger.info("Live telemetry WebSocket client connected: %s", websocket.client)

    try:
        while True:
            snapshot = collect_telemetry_snapshot()
            await websocket.send_text(json.dumps(snapshot))

            # Check for any inbound client ping/control messages without blocking
            try:
                # 1 second sampling interval
                await asyncio.wait_for(websocket.receive_text(), timeout=1.0)
            except asyncio.TimeoutError:
                # Standard polling cadence heartbeat
                pass
    except (WebSocketDisconnect, ConnectionResetError):
        logger.info("Live telemetry WebSocket client disconnected: %s", websocket.client)
    except Exception as exc:
        logger.warning("Error streaming telemetry over WebSocket: %s", exc)


@router.get("/telemetry/snapshot")
async def telemetry_snapshot_endpoint() -> JSONResponse:
    """REST endpoint providing the instantaneous telemetry snapshot for polling clients.

    Returns:
        JSONResponse containing current hardware and benchmark telemetry.
    """
    snapshot = collect_telemetry_snapshot()
    return JSONResponse(content=snapshot)
