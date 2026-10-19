# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Monitor FastAPI Application Facade

Direct interface to the benchmark telemetry FastAPI app, WebSocket routes, and SSE streaming handlers.
"""

from __future__ import annotations

from neuralops.runners.monitor.fastapi_app import *  # noqa: F403
import neuralops.runners.monitor.fastapi_app as _mod

__all__ = [name for name in dir(_mod) if not name.startswith("_")]
