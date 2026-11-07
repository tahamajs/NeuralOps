#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility re-export facade for GLUE benchmark dataset loading (F-61)."""

from __future__ import annotations

from typing import Any

from neuralops.data.glue_loader import (
    GLUE_LOCAL_DIR,
    GLUE_TASK_KEYS,
    get_num_labels_and_is_regression,
    load_glue_local,
)


def load_glue_dataset(*args: Any, **kwargs: Any) -> Any:
    """Load a GLUE dataset, delegating to canonical glue_loader."""
    return load_glue_local(*args, **kwargs)


__all__ = [
    "load_glue_dataset",
    "load_glue_local",
    "get_num_labels_and_is_regression",
    "GLUE_TASK_KEYS",
    "GLUE_LOCAL_DIR",
]
