#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for kernel autotuner."""
import time
from neuralops.core.triton_kernels.autotuner import Autotuner, autotune_kernel


def test_autotuner():
    configs = [
        {"block_size": 16, "speed": 0.002},
        {"block_size": 32, "speed": 0.0001},
        {"block_size": 64, "speed": 0.005},
    ]

    def runner(cfg):
        time.sleep(cfg["speed"])

    tuner = Autotuner(warmup=1, rep=2)
    best_cfg, best_time = tuner.tune(configs, runner)
    assert best_cfg["block_size"] == 32
    assert best_time > 0


def test_autotune_kernel_helper():
    configs = [{"k": 1, "s": 0.001}, {"k": 2, "s": 0.0001}]
    best = autotune_kernel(configs, lambda c: time.sleep(c["s"]))
    assert best["k"] == 2
