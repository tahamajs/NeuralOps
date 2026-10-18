"""
Benchmark script for FlashGQA: CUDA vs PyTorch Ref vs PyTorch SDPA
"""

import torch
from neuralops.core.cuda_kernels.flash_gqa import flash_gqa, flash_gqa_pytorch_ref


def run_benchmark():
    """Execute run benchmark routine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, H, H_KV, N, D = 4, 32, 8, 1024, 64
    print(f"Benchmarking FlashGQA with shape B={B}, H={H}, H_KV={H_KV}, N={N}, D={D} on {device}")

    torch.manual_seed(42)
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H_KV, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H_KV, N, D, device=device, dtype=torch.float32)

    assert callable(flash_gqa), "flash_gqa must be callable"
    assert callable(flash_gqa_pytorch_ref), "flash_gqa_pytorch_ref must be callable"

    # Warmup
    for _ in range(10):
        _ = flash_gqa(q, k, v, is_causal=False)
        _ = flash_gqa_pytorch_ref(q, k, v, is_causal=False)
    torch.cuda.synchronize()

    # Benchmark CUDA
    iters = 50
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        _ = flash_gqa(q, k, v, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    cuda_ms = start.elapsed_time(end) / iters

    # Benchmark PyTorch Ref
    start.record()
    for _ in range(iters):
        _ = flash_gqa_pytorch_ref(q, k, v, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    ref_ms = start.elapsed_time(end) / iters

    # Benchmark PyTorch SDPA (cuBLAS FlashAttention)
    k_rep = k.repeat_interleave(H // H_KV, dim=1)
    v_rep = v.repeat_interleave(H // H_KV, dim=1)
    for _ in range(10):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    torch.cuda.synchronize()

    start.record()
    for _ in range(iters):
        _ = torch.nn.functional.scaled_dot_product_attention(q, k_rep, v_rep, is_causal=False)
    end.record()
    torch.cuda.synchronize()
    sdpa_ms = start.elapsed_time(end) / iters

    print("Results:")
    print(f"  FlashGQA CUDA: {cuda_ms:.3f} ms")
    print(f"  PyTorch Ref:    {ref_ms:.3f} ms (Speedup vs Ref: {ref_ms / cuda_ms:.2f}x)")
    print(f"  PyTorch SDPA:   {sdpa_ms:.3f} ms (Speedup vs SDPA: {sdpa_ms / cuda_ms:.2f}x)")


if __name__ == "__main__":
    run_benchmark()
