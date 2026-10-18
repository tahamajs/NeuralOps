"""
Monitor FastAPI Application Facade

Direct interface to the benchmark telemetry FastAPI app, WebSocket routes, and SSE streaming handlers.

Author: Taha Majlesi
"""

from __future__ import annotations

from neuralops.runners.monitor.fastapi_app import *  # noqa: F403
import neuralops.runners.monitor.fastapi_app as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
