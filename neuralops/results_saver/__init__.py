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

from .config import SaveConfig
from .saver import ResultsSaver
from .storage.sqlite_ops import configure_sqlite_pragmas, sqlite_file_lock
from .writers.csv_writer import MASTER_CSV_COLUMNS, save_result
from .writers.json_parquet_writer import save_result_json, save_trajectories_parquet

__all__ = [
    "MASTER_CSV_COLUMNS",
    "ResultsSaver",
    "SaveConfig",
    "configure_sqlite_pragmas",
    "save_result",
    "save_result_json",
    "save_trajectories_parquet",
    "sqlite_file_lock",
]

from neuralops.results_saver import backends
from neuralops.results_saver import pipeline
from neuralops.results_saver import storage
from neuralops.results_saver import writers
from neuralops.results_saver import config
from neuralops.results_saver import saver
