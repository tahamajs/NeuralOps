#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Training Throughput and VRAM Peak Profiler for NeuralOps & FLA Models

Profiles forward+backward training step execution time, tokens-per-second, TFLOPS estimation, and peak allocated VRAM.
"""

from __future__ import annotations

import gc
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM


class TrainingThroughputProfiler:
    """Measures training step throughput, backward latency, and VRAM consumption."""

    def __init__(
        self,
        model: Union[str, nn.Module] = "gpt2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        dtype: torch.dtype = torch.float16 if torch.cuda.is_available() else torch.float32,
    ):
        self.device = device
        self.dtype = dtype

        if isinstance(model, str):
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=self.dtype,
            ).to(self.device).train()
        else:
            self.model = model.to(self.device).train()

    def profile_training_step(
        self,
        batch_size: int = 2,
        seq_len: int = 128,
        warmup_steps: int = 2,
        num_steps: int = 3,
        grad_accum_steps: int = 1,
    ) -> Dict[str, float]:
        """
        Profiles standard training step (fwd + bwd).
        Returns step_latency_ms, tokens_per_second, peak_vram_mb.
        """
        vocab_size = getattr(getattr(self.model, "config", None), "vocab_size", 1000)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(self.device)

        for _ in range(warmup_steps):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            labels = inputs.clone()
            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].sum()
            loss.backward()
            optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        step_times = []
        for _ in range(num_steps):
            optimizer.zero_grad(set_to_none=True)
            inputs = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
            labels = inputs.clone()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            outputs = self.model(inputs, labels=labels)
            loss = outputs.loss if hasattr(outputs, "loss") else outputs[0].sum()
            loss.backward()
            optimizer.step()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append((time.perf_counter() - t0) * 1000.0)

        avg_step_ms = sum(step_times) / len(step_times) if step_times else 0.0
        tokens_per_step = batch_size * seq_len
        tokens_per_sec = (tokens_per_step / (avg_step_ms / 1000.0)) if avg_step_ms > 0 else 0.0

        peak_vram_mb = 0.0
        if torch.cuda.is_available():
            peak_vram_mb = torch.cuda.max_memory_allocated(self.device) / (1024.0 * 1024.0)

        return {
            "batch_size": batch_size,
            "seq_len": seq_len,
            "step_latency_ms": round(avg_step_ms, 3),
            "tokens_per_second": round(tokens_per_sec, 2),
            "peak_vram_mb": round(peak_vram_mb, 2),
        }
