# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Runners Monitor Subpackage.
"""

from __future__ import annotations

from neuralops.runners.monitor.endpoints import (
    get_doc_content,
    get_loss_history,
    get_module_detail,
    launch_new_run,
    stop_current_run,
)
from neuralops.runners.monitor.rendering import (
    render_csv_export,
    render_favicon,
    render_json_export,
)
from neuralops.runners.monitor.server import (
    ComprehensiveStudioHandler,
    run_server,
)
from neuralops.runners.monitor.stats import (
    build_aggregated_stats,
    get_module_doc_index,
    load_harvested_benchmark_records,
)
from neuralops.runners.monitor.telemetry import (
    get_live_gpu_telemetry,
    get_live_logs,
)

__all__ = [
    "ComprehensiveStudioHandler",
    "run_server",
    "get_live_gpu_telemetry",
    "get_live_logs",
    "load_harvested_benchmark_records",
    "get_module_doc_index",
    "build_aggregated_stats",
    "get_loss_history",
    "get_doc_content",
    "stop_current_run",
    "launch_new_run",
    "get_module_detail",
    "render_favicon",
    "render_csv_export",
    "render_json_export",
]

from neuralops.runners.monitor import endpoints
from neuralops.runners.monitor import rendering
from neuralops.runners.monitor import server
from neuralops.runners.monitor import stats
from neuralops.runners.monitor import telemetry
