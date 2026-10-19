# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Project Path Utilities

Resolves standard filesystem paths, workspace locations, and checkpoint directories.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Returns the absolute path to the project root directory (uni/Project)."""
    return Path(__file__).resolve().parent.parent.parent.parent


def get_datasets_dir() -> Path:
    """Returns the absolute path to the global datasets directory."""
    return get_project_root() / "datasets"


def get_glue_dir() -> Path:
    """Returns the absolute path to the local GLUE datasets."""
    return get_datasets_dir() / "glue"


def resolve_path(relative_path: str) -> str:
    """Resolves a path relative to the project root and returns as string."""
    return str((get_project_root() / relative_path).resolve())
