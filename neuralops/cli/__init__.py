# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps CLI Subpackage.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "bench_cmd",
    "compress_cmd",
    "doctor",
    "eval_cmd",
    "info",
    "list_cmd",
    "serve_cmd",
    "test_cmd",
    "train_cmd",
    "terminal_ui",
]


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loading for neuralops.cli submodules."""
    try:
        module = importlib.import_module(f"neuralops.cli.{name}")
        globals()[name] = module
        return module
    except ModuleNotFoundError:
        pass
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
