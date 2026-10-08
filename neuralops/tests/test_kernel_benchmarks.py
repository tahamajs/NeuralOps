"""Tests for kernel benchmark harness."""
from neuralops.benchmarks.kernel_benchmark import benchmark_kernel, run_attention_kernel_benchmarks


def test_benchmark_kernel():
    t_ms = benchmark_kernel(lambda: [x * 2 for x in range(100)], warmup=1, repeat=2)
    assert t_ms >= 0.0


def test_run_attention_kernel_benchmarks():
    res = run_attention_kernel_benchmarks(batch_size=1, seq_len=16, head_dim=16, num_heads=2)
    assert "sdpa_ms" in res
    assert res["sdpa_ms"] > 0
    assert "tokens_per_sec" in res
