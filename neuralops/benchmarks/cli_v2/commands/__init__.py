"""Commands package for NeuralOps CLI v2."""

from __future__ import annotations

from neuralops.benchmarks.cli_v2.commands.benchmark import handle_benchmark, setup_benchmark_parser
from neuralops.benchmarks.cli_v2.commands.doctor import handle_doctor, setup_doctor_parser
from neuralops.benchmarks.cli_v2.commands.plan import handle_plan, setup_plan_parser
from neuralops.benchmarks.cli_v2.commands.results import handle_results, setup_results_parser
from neuralops.benchmarks.cli_v2.commands.ui import handle_ui, setup_ui_parser

__all__ = [
    "handle_benchmark",
    "handle_doctor",
    "handle_plan",
    "handle_results",
    "handle_ui",
    "setup_benchmark_parser",
    "setup_doctor_parser",
    "setup_plan_parser",
    "setup_results_parser",
    "setup_ui_parser",
]
