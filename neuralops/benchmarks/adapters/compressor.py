#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Benchmark adapter for NeuralOps PEFT/compression methods."""

from __future__ import annotations

import copy
import time
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn


class _PeftBenchmarkModel(nn.Module):
    """Small model whose projection names match the PEFT target contracts."""

    def __init__(self, hidden_size: int = 128) -> None:
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q, k, v = self.query(x), self.key(x), self.value(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
        return self.dense(torch.matmul(scores.softmax(dim=-1), v))


class CompressorModuleAdapter:
    """Measure real LoRA, PiSSA, and DoRA injection on a common model.

    ``param_reduction_pct`` is the reduction in trainable parameters relative
    to the original dense model.  The adapter also reports baseline-relative
    forward speed, making the comparison explicit for the lossy/PEFT cases.
    """

    METHODS = ("lora", "pissa", "dora")

    def __init__(self, device: str | torch.device | None = None, hidden_size: int = 128) -> None:
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.hidden_size = hidden_size

    def _timed(self, model: nn.Module, x: torch.Tensor, repeats: int = 10) -> float:
        model.eval()
        with torch.no_grad():
            for _ in range(2):
                model(x)
            if self.device.type == "cuda":
                start, end = torch.cuda.Event(True), torch.cuda.Event(True)
                start.record()
                for _ in range(repeats):
                    model(x)
                end.record()
                end.synchronize()
                return float(start.elapsed_time(end) / repeats)
            started = time.perf_counter()
            for _ in range(repeats):
                model(x)
            return (time.perf_counter() - started) * 1000.0 / repeats

    def _inject(self, name: str, model: nn.Module) -> int:
        if name == "lora":
            from neuralops.trainer.methods.lora import apply_lora

            apply_lora(model, SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0))
            return sum(hasattr(module, "lora_A") for module in model.modules())
        if name == "dora":
            from neuralops.trainer.methods.lora import apply_dora

            apply_dora(model, SimpleNamespace(lora_r=4, lora_alpha=8.0, lora_dropout=0.0))
            return sum(hasattr(module, "lora_A") for module in model.modules())
        if name == "pissa":
            from neuralops.trainer.peft_recovery.svd_adapters import inject_pissa_adapters

            return int(
                inject_pissa_adapters(
                    model,
                    rank=4,
                    target_modules=["query", "key", "value", "dense"],
                )
            )
        raise ValueError(f"Unsupported PEFT method: {name}")

    def benchmark_module(self, module_name: str) -> dict[str, Any]:
        """Return measured latency, trainable reduction, and finite-output status."""
        if module_name not in self.METHODS:
            raise ValueError(f"module_name must be one of {self.METHODS}")
        torch.manual_seed(42)
        base = _PeftBenchmarkModel(self.hidden_size).to(self.device)
        candidate = copy.deepcopy(base)
        x = torch.randn(2, 32, self.hidden_size, device=self.device)
        baseline_ms = self._timed(base, x)
        injected = self._inject(module_name, candidate)
        candidate_ms = self._timed(candidate, x)
        with torch.no_grad():
            output = candidate(x)
        original_params = sum(parameter.numel() for parameter in base.parameters())
        trainable_params = sum(parameter.numel() for parameter in candidate.parameters() if parameter.requires_grad)
        reduction = 100.0 * (1.0 - trainable_params / max(original_params, 1))
        status = "PASS_FAST" if baseline_ms / candidate_ms >= 1.05 else "PASS_SLOW"
        if not torch.isfinite(output).all():
            status = "FAIL"
        return {
            "module": module_name,
            "category": "peft",
            "cuda_latency_ms": candidate_ms,
            "pytorch_latency_ms": baseline_ms,
            "speedup": baseline_ms / candidate_ms if candidate_ms > 0 else 0.0,
            "param_reduction_pct": reduction,
            "trainable_params": trainable_params,
            "injected_modules": injected,
            "peak_vram_mb": torch.cuda.max_memory_allocated(self.device) / (1024**2) if self.device.type == "cuda" else 0.0,
            "status": status,
        }


__all__ = ["CompressorModuleAdapter"]
