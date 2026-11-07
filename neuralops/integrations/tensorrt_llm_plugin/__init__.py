#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
TensorRT-LLM Plugin Module for NeuralOps Attention.
"""

from __future__ import annotations

from neuralops.integrations.tensorrt_llm_plugin.plugin import neuralops_attention, python_plugin

__all__ = [
    "neuralops_attention",
    "python_plugin",
]
