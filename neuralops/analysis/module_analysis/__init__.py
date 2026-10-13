"""Module analysis package for NeuralOps."""

from __future__ import annotations

from neuralops.analysis.module_analysis.aggregators import (
    determine_fix_priority,
    determine_sub_category,
    get_all_module_rows,
    get_best_dataset_result,
)
from neuralops.analysis.module_analysis.constants import (
    ALIASES,
    CSV_OUT,
    KNOWN_LIT,
    MD_OUT,
    PAPERS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
)
from neuralops.analysis.module_analysis.main import main, run_all_modules_analysis
from neuralops.analysis.module_analysis.parsers import (
    load_all_csv_records,
    resolve_code_path,
)
from neuralops.analysis.module_analysis.reporters import (
    ANALYSIS_FIELDNAMES,
    write_csv_report,
    write_markdown_report,
)

__all__ = [
    "PROJECT_ROOT",
    "PAPERS_DIR",
    "RESULTS_DIR",
    "CSV_OUT",
    "MD_OUT",
    "KNOWN_LIT",
    "ALIASES",
    "load_all_csv_records",
    "resolve_code_path",
    "get_all_module_rows",
    "get_best_dataset_result",
    "determine_sub_category",
    "determine_fix_priority",
    "ANALYSIS_FIELDNAMES",
    "write_csv_report",
    "write_markdown_report",
    "run_all_modules_analysis",
    "main",
]

# Auto-wired package members
from neuralops.analysis.module_analysis import aggregators
from neuralops.analysis.module_analysis import constants
from neuralops.analysis.module_analysis import main
from neuralops.analysis.module_analysis import parsers
from neuralops.analysis.module_analysis import reporters
