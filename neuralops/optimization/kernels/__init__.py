#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Optimization Kernels Facade.

Canonical Owner: neuralops.ops
"""

from __future__ import annotations

from typing import Any

import neuralops.ops as _ops


def __getattr__(name: str) -> Any:
    return getattr(_ops, name)


def __dir__() -> list[str]:
    return dir(_ops)


__all__ = ["_ops"]
