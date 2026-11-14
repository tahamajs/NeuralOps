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
        """Run benchmark iterations and compute step time and throughput."""
        model.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        def _forward_step() -> None:
            out = model(**batch)
            if hasattr(out, "loss") and out.loss is not None:
                loss = out.loss
            elif isinstance(out, (tuple, list)) and len(out) > 0:
                loss = out[0]
            elif isinstance(out, torch.Tensor):
                loss = out
            else:
                loss = getattr(out, "logits", out)
            if isinstance(loss, torch.Tensor) and loss.requires_grad:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        for _ in range(warmup):
            _forward_step()

        is_cuda = torch.cuda.is_available() and str(device).startswith("cuda")

        if is_cuda:
            dev = torch.device(device)
            # Rule 2.3 — L2 Cache Flush before timing
            try:
                l2_bytes = torch.cuda.get_device_properties(dev).L2_cache_size
                buffer = torch.empty(l2_bytes // 4, dtype=torch.int32, device=dev)
                buffer.zero_()
                torch.cuda.synchronize(dev)
            except Exception:
                pass

            # Rule 2.2 — CUDA Timing with Events
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            for _ in range(iters):
                _forward_step()
            end_event.record()
            torch.cuda.synchronize(dev)
            elapsed_ms = start_event.elapsed_time(end_event)
            dt = (elapsed_ms / 1000.0) / max(1, iters)
        else:
            t0 = time.perf_counter()
            for _ in range(iters):
                _forward_step()
            dt = (time.perf_counter() - t0) / max(1, iters)

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
