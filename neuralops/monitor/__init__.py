#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Monitoring, Telemetry, and Web Dashboard Suite.
"""

from __future__ import annotations

from neuralops.runners.monitor import (
    ComprehensiveStudioHandler,
    build_aggregated_stats,
    get_doc_content,
    get_live_gpu_telemetry,
    get_live_logs,
    get_loss_history,
    get_module_detail,
    get_module_doc_index,
    launch_new_run,
    load_harvested_benchmark_records,
    render_csv_export,
    render_favicon,
    render_json_export,
    run_server,
    stop_current_run,
)

__all__ = [
    "ComprehensiveStudioHandler",
    "build_aggregated_stats",
    "get_doc_content",
    "get_live_gpu_telemetry",
    "get_live_logs",
    "get_loss_history",
    "get_module_detail",
    "get_module_doc_index",
    "launch_new_run",
    "load_harvested_benchmark_records",
    "render_csv_export",
    "render_favicon",
    "render_json_export",
    "run_server",
    "stop_current_run",
]
