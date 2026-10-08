"""Tests for benchmark runner suite."""
import os
import torch
from neuralops.benchmarks.suite_runner import (
    benchmark_attention_speed,
    benchmark_attention_memory,
    benchmark_attention_accuracy,
    benchmark_scaling_sweep,
    run_and_save_benchmark_suite,
)


def test_speed_and_memory_benchmark():
    x = torch.randn(10, 10)
    ms = benchmark_attention_speed(lambda: x @ x, warmup=1, repeat=2)
    assert ms > 0.0

    mem = benchmark_attention_memory(lambda: x @ x)
    assert mem >= 0.0


def test_accuracy_benchmark():
    a = torch.tensor([1.0, 2.0])
    b = torch.tensor([1.0, 2.01])
    res = benchmark_attention_accuracy(a, b)
    assert res["max_abs_err"] < 0.02
    assert res["rel_frob_err"] < 0.02


def test_scaling_and_save_suite():
    res = benchmark_scaling_sweep(seq_lens=[16, 32], batch_size=1, num_heads=2, head_dim=8)
    assert len(res) == 2

    csv_path, md_path = run_and_save_benchmark_suite(output_dir="/tmp/bench_test")
    assert os.path.exists(csv_path)
    assert os.path.exists(md_path)
