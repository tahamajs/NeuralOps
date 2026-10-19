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

from neuralops.core.registry import COMPRESSOR_REGISTRY
from neuralops.core.base import BaseCompressor
from neuralops.core.compression_pipeline import compress_model
import neuralops.compression_engine as _comp_engine

__all__ = ["COMPRESSOR_REGISTRY", "BaseCompressor", "compress_model"]
