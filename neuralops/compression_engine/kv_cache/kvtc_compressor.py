#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Backward-compatibility shim for neuralops.compression.kv_cache.kvtc_compressor."""

from __future__ import annotations

from neuralops.compression.kv_cache.kvtc_compressor import *  # noqa: F401, F403
