#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Hardware Configuration

Settings for device allocation, precision targeting, and GPU execution environments.
"""

from dataclasses import dataclass


@dataclass
class HardwareConfig:
    device: str = "auto"
    force_cpu_for_svd: bool = True
    enable_fp16: bool = False
    enable_bf16: bool = False
    num_workers: int = 4
    pin_memory: bool = False
    torch_compile: bool = False
    empty_cache_every_step: bool = False
