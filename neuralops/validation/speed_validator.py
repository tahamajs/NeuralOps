#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Speed and Throughput Validator.

Benchmarks training throughput, iteration speed, and execution latency across hardware.
Returns structured ValidationStatus without stdout printing (F-57, F-97).
"""

from __future__ import annotations

import time
from typing import Any, Dict

import torch

from neuralops.validation.training_validator import ValidationStatus


class SpeedValidator:
    """Speed, latency, and throughput benchmark validator returning structured status."""

    @staticmethod
    def benchmark(
        model: torch.nn.Module,
        batch: Dict[str, Any],
        device: str = "cpu",
        warmup: int = 3,
        iters: int = 10,
    ) -> ValidationStatus:
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        for _ in range(warmup):
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(iters):
            out = model(**batch)
            loss = out.loss if hasattr(out, "loss") else out[0]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if torch.cuda.is_available() and device.startswith("cuda"):
            torch.cuda.synchronize()
        dt = (time.time() - t0) / iters
        bs = batch["input_ids"].size(0) if ("input_ids" in batch and hasattr(batch["input_ids"], "size")) else 1
        throughput = (bs / dt) if dt > 0 else 0.0

        return ValidationStatus({
            "status": "passed",
            "passed": True,
            "step_time_ms": dt * 1000.0,
            "throughput_samples_per_sec": throughput,
            "dt": dt,
            "iters": iters,
            "batch_size": bs,
        })


__all__ = ["SpeedValidator", "ValidationStatus"]
