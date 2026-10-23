#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
NeuralOps Unified Compression Engines.
"""

from __future__ import annotations

from typing import Any

from neuralops.core.registry import COMPRESSOR_REGISTRY, get_compressor, list_compressors
from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
import neuralops.compression_engine as _comp_engine


def __getattr__(name: str) -> Any:
    return getattr(_comp_engine, name)


def __dir__() -> list[str]:
    return list(_comp_engine.__dir__()) + ["COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model", "get_compressor", "list_compressors"]


__all__ = sorted(list(set(__dir__())))
