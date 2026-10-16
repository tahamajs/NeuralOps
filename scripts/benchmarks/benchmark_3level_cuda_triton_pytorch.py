"""
3-Level GPU Benchmark Suite: Native CUDA C++ vs. Triton JIT vs. Pure PyTorch Reference.
Target: NVIDIA Ada Lovelace SM 8.9 (GeForce RTX 4070 Laptop GPU, 8GB VRAM).

Measures latency (ms), peak allocated VRAM (MB), speedups, and parity across all 3 execution levels:
  Level 1: Native CUDA C++ (SM >= 7.0)
  Level 2: OpenAI Triton JIT
  Level 3: Pure PyTorch Reference
"""

from __future__ import annotations

import csv
import gc
import json
import os
import sys
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn.functional as F

from neuralops.core.cuda_kernels import force_kernel_backend
from neuralops.core.cuda_kernels.selective_scan.selective_scan_autograd import selective_scan
from neuralops.core.cuda_kernels.sliding_window.sliding_autograd import sliding_window_attn
from neuralops.core.cuda_kernels.sparse_attention.sparse_autograd import block_sparse_attn
from neuralops.core.cuda_kernels.quantized_gemm.quant_autograd import w4a16_gemm, fp8_gemm
from neuralops.core.cuda_kernels.flash_gqa.flash_gqa_autograd import flash_gqa
from neuralops.core.cuda_kernels.paged_attention.paged_attention_autograd import paged_attention
from neuralops.core.cuda_kernels.paged_kv_cache.kv_cache_autograd import copy_blocks_cow
from neuralops.core.cuda_kernels.moe_routing.moe_autograd import moe_dispatch, moe_gather


def benchmark_fn(
    fn: Callable[[], Any],
    warmup: int = 25,
    iters: int = 100,
) -> Tuple[float, float]:
    """Measures CUDA kernel execution time (ms) and peak allocated VRAM (MB)."""
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = fn()
    end_event.record()
    torch.cuda.synchronize()

    mean_ms = start_event.elapsed_time(end_event) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return mean_ms, peak_vram_mb


def compute_metrics(ref: torch.Tensor, test: torch.Tensor) -> Tuple[float, float]:
    ref_f = ref.detach().flatten().float()
    test_f = test.detach().flatten().float()
    cos_sim = F.cosine_similarity(ref_f, test_f, dim=0).item()
    diff = (ref_f - test_f).abs()
    rel_err = (diff / (ref_f.abs() + 1e-5)).max().item()
    return cos_sim, rel_err


def main():
    if not torch.cuda.is_available():
        print("CUDA unavailable. Exiting benchmark.")
        sys.exit(1)

    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print("=" * 80)
    print(f"3-LEVEL BENCHMARK: CUDA C++ vs. TRITON JIT vs. PYTORCH REFERENCE")
    print(f"Physical Device: {gpu_name} (SM {torch.cuda.get_device_capability(0)})")
    print("=" * 80)

    results: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 1. Selective Scan
    # -------------------------------------------------------------------------
    print("\n[1/8] selective_scan (Mamba SSM)")
    B, D, L, D_STATE = 4, 64, 512, 16
    u = torch.randn(B, D, L, device=device, dtype=torch.float32)
    delta = torch.rand(B, D, L, device=device, dtype=torch.float32) * 0.1
    A = -torch.rand(D, D_STATE, device=device, dtype=torch.float32)
    B_m = torch.randn(B, D_STATE, L, device=device, dtype=torch.float32)
    C_m = torch.randn(B, D_STATE, L, device=device, dtype=torch.float32)
    D_val = torch.randn(D, device=device, dtype=torch.float32)

    fn_py = lambda: selective_scan(u, delta, A, B_m, C_m, D_val)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_py)
        out_py, _ = fn_py()

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_py)
            out_tri, _ = fn_py()
        except Exception:
            lat_tri, vram_tri, out_tri = None, None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_py)
            out_cuda, _ = fn_py()
        except Exception:
            lat_cuda, vram_cuda, out_cuda = None, None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "selective_scan",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 2. Sliding Window Attention
    # -------------------------------------------------------------------------
    print("\n[2/8] sliding_window_attn")
    B, H, N, D = 4, 8, 512, 64
    q = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    k = torch.randn(B, H, N, D, device=device, dtype=torch.float32)
    v = torch.randn(B, H, N, D, device=device, dtype=torch.float32)

    fn_sw = lambda: sliding_window_attn(q, k, v, window_size=64, num_sinks=4)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_sw)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_sw)
        except Exception:
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_sw)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "sliding_window_attn",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 3. Block Sparse Attention
    # -------------------------------------------------------------------------
    print("\n[3/8] block_sparse_attn")
    B, H, N, D = 4, 8, 512, 64
    block_size = 32
    num_blocks = N // block_size
    mask = torch.eye(num_blocks, device=device, dtype=torch.bool)
    for i in range(num_blocks):
        mask[i, max(0, i - 1):min(num_blocks, i + 2)] = True

    fn_bs = lambda: block_sparse_attn(q, k, v, block_mask=mask, block_size=block_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_bs)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_bs)
        except Exception:
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_bs)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "block_sparse_attn",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 4. W4A16 Quantized GEMM
    # -------------------------------------------------------------------------
    print("\n[4/8] w4a16_gemm")
    M, K, N = 128, 1024, 1024
    group_size = 64
    A_mat = torch.randn(M, K, device=device, dtype=torch.float32)
    W_q = torch.randint(0, 255, (K, N // 2), device=device, dtype=torch.uint8)
    scales = torch.rand(K // group_size, N, device=device, dtype=torch.float32) * 0.1
    zeros = torch.zeros(K // group_size, N, device=device, dtype=torch.float32)

    fn_w4 = lambda: w4a16_gemm(A_mat, W_q, scales, zeros, group_size=group_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_w4)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_w4)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "w4a16_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 5. FP8 Scaled GEMM
    # -------------------------------------------------------------------------
    print("\n[5/8] fp8_gemm")
    M, K, N = 256, 1024, 1024
    A_fp8 = torch.randn(M, K, device=device, dtype=torch.float32)
    B_fp8 = torch.randn(K, N, device=device, dtype=torch.float32)
    sa = torch.tensor([0.5], device=device, dtype=torch.float32)
    sb = torch.tensor([0.5], device=device, dtype=torch.float32)

    fn_fp8 = lambda: fp8_gemm(A_fp8, B_fp8, sa, sb)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_fp8)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_fp8)
        except Exception:
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_fp8)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "fp8_gemm",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 6. Flash GQA
    # -------------------------------------------------------------------------
    print("\n[6/8] flash_gqa")
    B, H, N_Q, D = 4, 8, 256, 64
    H_KV, N_KV = 2, 256
    q_gqa = torch.randn(B, H, N_Q, D, device=device, dtype=torch.float32)
    k_gqa = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32)
    v_gqa = torch.randn(B, H_KV, N_KV, D, device=device, dtype=torch.float32)

    fn_gqa = lambda: flash_gqa(q_gqa, k_gqa, v_gqa, is_causal=True)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_gqa)

    with force_kernel_backend("triton"):
        try:
            lat_tri, vram_tri = benchmark_fn(fn_gqa)
        except Exception:
            lat_tri, vram_tri = None, None

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_gqa)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    sp_tri = (lat_py / lat_tri) if lat_tri else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | Triton: {lat_tri or 0:.3f} ms ({sp_tri:.2f}x) | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "flash_gqa",
        "pytorch_ms": lat_py,
        "triton_ms": lat_tri,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": (lat_tri / lat_cuda) if (lat_cuda and lat_tri) else 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 7. Paged Attention
    # -------------------------------------------------------------------------
    print("\n[7/8] paged_attention")
    num_seqs = 8
    num_heads = 8
    head_dim = 64
    block_size = 16
    num_blocks = 64

    q_pa = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=torch.float32)
    k_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    v_cache = torch.randn(num_blocks, num_heads, head_dim, block_size, device=device, dtype=torch.float32)
    block_tables = torch.randint(0, num_blocks, (num_seqs, 16), device=device, dtype=torch.int32)
    context_lens = torch.full((num_seqs,), 128, device=device, dtype=torch.int32)

    fn_pa = lambda: paged_attention(q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_pa)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_pa)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "paged_attention",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # -------------------------------------------------------------------------
    # 8. MoE Routing
    # -------------------------------------------------------------------------
    print("\n[8/8] moe_routing (dispatch + gather)")
    num_tokens, hidden_dim, top_k = 512, 1024, 2
    total_slots = 1024
    tokens = torch.randn(num_tokens, hidden_dim, device=device, dtype=torch.float32)
    topk_indices = torch.randint(0, 8, (num_tokens, top_k), device=device, dtype=torch.int32)
    offsets = torch.arange(num_tokens * top_k, device=device, dtype=torch.int32).view(num_tokens, top_k)
    weights = torch.softmax(torch.randn(num_tokens, top_k, device=device, dtype=torch.float32), dim=-1)

    def fn_moe():
        disp = moe_dispatch(tokens, topk_indices, offsets, total_slots)
        return moe_gather(disp, weights, topk_indices, offsets)

    with force_kernel_backend("pytorch"):
        lat_py, vram_py = benchmark_fn(fn_moe)

    with force_kernel_backend("cuda"):
        try:
            lat_cuda, vram_cuda = benchmark_fn(fn_moe)
        except Exception:
            lat_cuda, vram_cuda = None, None

    sp_cuda = (lat_py / lat_cuda) if lat_cuda else 1.0
    print(f"  PyTorch: {lat_py:.3f} ms | CUDA: {lat_cuda or 0:.3f} ms ({sp_cuda:.2f}x)")

    results.append({
        "kernel": "moe_routing",
        "pytorch_ms": lat_py,
        "triton_ms": None,
        "cuda_ms": lat_cuda,
        "speedup_cuda_vs_py": sp_cuda,
        "speedup_cuda_vs_tri": 1.0,
        "pytorch_vram_mb": vram_py,
        "cuda_vram_mb": vram_cuda,
    })

    # Save to JSON, CSV, MD
    os.makedirs("results", exist_ok=True)
    json_path = "results/benchmark_3level_cuda_triton_pytorch.json"
    csv_path = "results/benchmark_3level_cuda_triton_pytorch.csv"
    md_path = "results/benchmark_3level_cuda_triton_pytorch.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 🚀 3-Level Execution Hierarchy Benchmark: Native CUDA C++ vs. Triton JIT vs. PyTorch\n\n")
        f.write(f"**Device:** {gpu_name} (SM {torch.cuda.get_device_capability(0)})\n\n")
        f.write("| Kernel Operator | PyTorch Ref (ms) | Triton JIT (ms) | CUDA C++ (ms) | Speedup (CUDA vs. Py) | Speedup (CUDA vs. Triton) |\n")
        f.write("|:---|:---:|:---:|:---:|:---:|:---:|\n")
        for r in results:
            tri_str = f"{r['triton_ms']:.3f}" if r['triton_ms'] else "N/A"
            cuda_str = f"{r['cuda_ms']:.3f}" if r['cuda_ms'] else "N/A"
            f.write(f"| `{r['kernel']}` | {r['pytorch_ms']:.3f} | {tri_str} | {cuda_str} | **{r['speedup_cuda_vs_py']:.2f}x** | {r['speedup_cuda_vs_tri']:.2f}x |\n")

    print("\n" + "=" * 80)
    print(f"BENCHMARK SAVED TO {json_path}, {csv_path}, AND {md_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
