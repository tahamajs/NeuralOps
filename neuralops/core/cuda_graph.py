# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
CUDA Graph Execution Engine

CUDA Graph runner capturing and replaying static-shape execution graphs for latency-critical inference.
"""

from __future__ import annotations
from typing import Callable
import torch

__all__ = ["CUDAGraphRunner"]


class CUDAGraphRunner:
    """Captures and replays static computation graph via CUDA Graphs with CPU fallback."""

    def __init__(self, fn: Callable[..., torch.Tensor], warmup: int = 3) -> None:
        self.fn = fn
        self.warmup = warmup
        self.graph: torch.cuda.CUDAGraph | None = None
        self.static_inputs: list[torch.Tensor] = []
        self.static_output: torch.Tensor | None = None

    def capture(self, *sample_inputs: torch.Tensor) -> None:
        """Captures static execution graph on CUDA device."""
        if not torch.cuda.is_available() or sample_inputs[0].device.type != "cuda":
            return

        device = sample_inputs[0].device
        s = torch.cuda.Stream(device=device)
        s.wait_stream(torch.cuda.current_stream(device=device))
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                _ = self.fn(*sample_inputs)
        torch.cuda.current_stream(device=device).wait_stream(s)

        self.static_inputs = [x.clone() for x in sample_inputs]
        self.graph = torch.cuda.CUDAGraph()

        with torch.cuda.graph(self.graph):
            self.static_output = self.fn(*self.static_inputs)

    def replay(self, *inputs: torch.Tensor) -> torch.Tensor:
        """Replays graph if captured; falls back to fn(*inputs) otherwise."""
        if self.graph is None or not torch.cuda.is_available():
            return self.fn(*inputs)

        for static_in, dynamic_in in zip(self.static_inputs, inputs):
            static_in.copy_(dynamic_in)

        self.graph.replay()
        return self.static_output.clone()
