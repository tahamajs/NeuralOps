"""Tests for CUDAGraphRunner."""
import torch
from neuralops.core.cuda_graph import CUDAGraphRunner


def test_cuda_graph_runner_cpu_fallback():
    def add_one(x):
        return x + 1.0

    runner = CUDAGraphRunner(add_one)
    x = torch.tensor([1.0, 2.0, 3.0])
    runner.capture(x)
    out = runner.replay(x)
    assert torch.allclose(out, torch.tensor([2.0, 3.0, 4.0]))
