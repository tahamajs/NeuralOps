#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Results Saver Subpackage.
"""

from .artifacts import RunArtifactManager
from .config import SaveConfig
from .engine import DualCommitPersistenceEngine
from .saver import MultiFormatSaver, ResultsSaver
from .storage.maintenance import SQLiteMaintenanceEngine
from .storage.sqlite_backend import SQLiteMasterLedger, init_master_ledger
from .storage.sqlite_ops import configure_sqlite_pragmas, sqlite_file_lock
from .writers.csv_writer import MASTER_CSV_COLUMNS, save_result
from .writers.json_parquet_writer import save_result_json, save_trajectories_parquet
from .writers.parquet_writer import read_metrics_parquet, write_metrics_parquet
from .writers.rolling_aggregator import RollingSuiteAggregator
from .writers.summary_generator import BenchmarkSummaryGenerator

__all__ = [
    "BenchmarkSummaryGenerator",
    "DualCommitPersistenceEngine",
    "MASTER_CSV_COLUMNS",
    "MultiFormatSaver",
    "ResultsSaver",
    "RollingSuiteAggregator",
    "RunArtifactManager",
    "SQLiteMaintenanceEngine",
    "SQLiteMasterLedger",
    "SaveConfig",
    "configure_sqlite_pragmas",
    "init_master_ledger",
    "read_metrics_parquet",
    "save_result",
    "save_result_json",
    "save_trajectories_parquet",
    "sqlite_file_lock",
    "write_metrics_parquet",
]

from neuralops.results_saver import artifacts
from neuralops.results_saver import backends
from neuralops.results_saver import config
from neuralops.results_saver import engine
from neuralops.results_saver import pipeline
from neuralops.results_saver import saver
from neuralops.results_saver import storage
from neuralops.results_saver import writers
