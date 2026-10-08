#!/usr/bin/env python3
"""Production-grade NeuralOps Telemetry, Benchmark Database & Documentation Studio Server.

Backward-compatibility shim. Logic has been decomposed into `neuralops.runners.monitor`.
"""

from __future__ import annotations

import os
from pathlib import Path

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

PORT = 8080
PROJECT_DIR = os.environ.get("PROJECT_DIR", str(Path(__file__).resolve().parent.parent))
IS_LOCAL_GPU = os.path.exists(r"D:\SVD_Project")
WEB_DIR = os.path.join(PROJECT_DIR, "Docs")
HARVESTED_JSON = os.path.join(PROJECT_DIR, "results", "harvested_benchmark_data.json")
DOCS_DIR = os.path.join(PROJECT_DIR, "Docs")

ComprehensiveStudioHandler.project_dir = PROJECT_DIR
ComprehensiveStudioHandler.web_dir = WEB_DIR
ComprehensiveStudioHandler.docs_dir = DOCS_DIR
ComprehensiveStudioHandler.harvested_json = HARVESTED_JSON
ComprehensiveStudioHandler.is_local_gpu = IS_LOCAL_GPU


def main() -> None:
    """CLI entry point."""
    run_server(PORT)


if __name__ == "__main__":
    main()

__all__ = [
    "PORT",
    "PROJECT_DIR",
    "IS_LOCAL_GPU",
    "WEB_DIR",
    "HARVESTED_JSON",
    "DOCS_DIR",
    "ComprehensiveStudioHandler",
    "get_live_gpu_telemetry",
    "load_harvested_benchmark_records",
    "get_module_doc_index",
    "build_aggregated_stats",
    "get_live_logs",
    "get_loss_history",
    "get_module_detail",
    "stop_current_run",
    "launch_new_run",
    "get_doc_content",
    "render_favicon",
    "render_csv_export",
    "render_json_export",
    "run_server",
    "main",
]
