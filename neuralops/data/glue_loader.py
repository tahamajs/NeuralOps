#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Load locally cached GLUE task datasets and expose task metadata."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

GLUE_LOCAL_DIR = Path(os.environ.get("NEURALOPS_GLUE_LOCAL_DIR", "data/glue"))
GLUE_TASK_KEYS: dict[str, tuple[str, str | None]] = {
    "cola": ("sentence", None),
    "sst2": ("sentence", None),
    "mrpc": ("sentence1", "sentence2"),
    "qqp": ("question1", "question2"),
    "stsb": ("sentence1", "sentence2"),
    "mnli": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "rte": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


def get_num_labels_and_is_regression(task: str) -> tuple[int, bool]:
    """Return the GLUE label count and whether the task is regression."""
    key = task.lower().replace("glue_", "")
    return (1, True) if key == "stsb" else (3, False) if key == "mnli" else (2, False)


def load_glue_local(task: str, base_path: str | Path | None = None) -> Any:
    """Load a cached task directory, falling back to the Hugging Face GLUE dataset."""
    key = task.lower().replace("glue_", "")
    if key not in GLUE_TASK_KEYS:
        raise ValueError(f"Unsupported GLUE task: {task}")
    root = Path(base_path) if base_path is not None else GLUE_LOCAL_DIR
    task_path = root / key
    import datasets

    if task_path.is_dir():
        return datasets.load_from_disk(str(task_path))
    try:
        return datasets.load_dataset("glue", key)
    except Exception as exc:
        raise RuntimeError(
            f"GLUE task '{key}' is not cached at {task_path} and could not be downloaded"
        ) from exc


__all__ = ["GLUE_LOCAL_DIR", "GLUE_TASK_KEYS", "get_num_labels_and_is_regression", "load_glue_local"]
