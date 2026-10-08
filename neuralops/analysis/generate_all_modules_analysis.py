#!/usr/bin/env python3
"""
Comprehensive 217-Module Analysis Generator.
Extracts and validates all 25 required columns for paper/_ALL_MODULES_ANALYSIS.csv
and paper/_ALL_MODULES_ANALYSIS.md according to project guidelines.

NOTE: This module is maintained as a backward-compatible façade for
`neuralops.analysis.module_analysis`.
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
