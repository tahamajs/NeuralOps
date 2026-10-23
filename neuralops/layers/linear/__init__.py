#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Linear attention layers."""

from __future__ import annotations

from typing import Any
import neuralops.attentions.linear as _linear


def __getattr__(name: str) -> Any:
    return getattr(_linear, name)


def __dir__() -> list[str]:
    return list(_linear.__all__)


__all__ = list(_linear.__all__)
