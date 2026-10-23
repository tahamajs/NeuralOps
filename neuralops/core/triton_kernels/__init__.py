#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.ops.triton."""

from __future__ import annotations

from typing import Any
import neuralops.ops.triton as _ops_triton


def __getattr__(name: str) -> Any:
    return getattr(_ops_triton, name)


def __dir__() -> list[str]:
    return list(_ops_triton.__dir__())
