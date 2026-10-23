#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""tensor_decomp compression engines."""

from __future__ import annotations

from typing import Any
import neuralops.compression_engine.tensor_decomp as _fam_mod


def __getattr__(name: str) -> Any:
    return getattr(_fam_mod, name)


def __dir__() -> list[str]:
    return list(_fam_mod.__all__)


__all__ = list(_fam_mod.__all__)
