"""
NeuralOps Monitoring, Telemetry, and Web Dashboard Suite.

Author: Taha Majlesi
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
