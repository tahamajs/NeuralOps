#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compatibility entry point for GLUE benchmark dataset loading."""

from typing import Any


def load_glue_dataset(*args: Any, **kwargs: Any) -> Any:
    """Load a GLUE dataset through the benchmark loader on demand."""
    from neuralops.benchmarks.core.data_loader import load_glue_dataset as _load

    return _load(*args, **kwargs)

__all__ = ["load_glue_dataset"]
