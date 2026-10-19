#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
All-Modules Analysis Generator

Automated analysis runner computing global benchmarks and aggregations across all registered attention modules.
"""

from __future__ import annotations

from neuralops.analysis.module_analysis import (
    ALIASES,
    ANALYSIS_FIELDNAMES,
    CSV_OUT,
    KNOWN_LIT,
    MD_OUT,
    PAPERS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    determine_fix_priority,
    determine_sub_category,
    get_all_module_rows,
    get_best_dataset_result,
    load_all_csv_records,
    main,
    resolve_code_path,
    run_all_modules_analysis,
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

if __name__ == "__main__":
    main()
