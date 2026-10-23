#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward compatibility shim for neuralops.compression."""

from __future__ import annotations

from typing import Any
import neuralops.compression as _mod

COMPRESSOR_REGISTRY = _mod.COMPRESSOR_REGISTRY
get_compressor = _mod.get_compressor
list_compressors = _mod.list_compressors
BaseCompressor = _mod.BaseCompressor
compress_model = _mod.compress_model

_SUBPACKAGES = _mod._SUBPACKAGES


def __getattr__(name: str) -> Any:
    return getattr(_mod, name)


def __dir__() -> list[str]:
    return list(_mod.__dir__())


__all__ = list(_mod.__all__)
