#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Benchmarks Data Subpackage.
"""

from neuralops.benchmarks.data.loader import (
    CSVDatasetLoader,
    DATASET_LOADER_REGISTRY,
    DatasetLoader,
    JSONDatasetLoader,
    ParquetDatasetLoader,
    auto_detect_columns,
    detect_dataset_task,
    load_universal_dataset,
    make_universal_dataloader,
    register_dataset_loader,
)

__all__ = [
    "DatasetLoader",
    "register_dataset_loader",
    "detect_dataset_task",
    "auto_detect_columns",
    "load_universal_dataset",
    "make_universal_dataloader",
    "CSVDatasetLoader",
    "JSONDatasetLoader",
    "ParquetDatasetLoader",
    "DATASET_LOADER_REGISTRY",
]

from neuralops.benchmarks.data import loader
