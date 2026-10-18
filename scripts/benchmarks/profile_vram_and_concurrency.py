"""Profile vram and concurrency module.

Author: Taha Majlesi
"""

import os
import gc
import json
import time
import torch
from neuralops.attentions import (
    GLAttention,
    CosformerAttention,
    GatedDeltaNet,
    SVDLinearAttention,
    TritonCosformerAttention,
    TritonSVDLinearAttention,
    MambaBlock,
)


class DummyConfig:
    """Dummyconfig definition."""

    hidden_size = 768
    num_attention_heads = 12
    num_heads = 12
    head_dim = 64
    d_model = 768


def get_cuda_mem():
    """Execute get cuda mem routine."""
    return {
        "allocated_mb": torch.cuda.memory_allocated() / (1024**2),
        "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024**2),
        "reserved_mb": torch.cuda.memory_reserved() / (1024**2),
        "max_reserved_mb": torch.cuda.max_memory_reserved() / (1024**2),
    }


def run_stress_test(name, module_fn, num_iters=1000, batch_size=4, seq_len=256, dim=768, warmup=20):
    """Execute run stress test routine."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    device = torch.device("cuda:0")
    module = module_fn().to(device)
    module.train()

    # Warmup
    for _ in range(warmup):
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        out = module(x)
        y = out[0] if isinstance(out, (tuple, list)) else out
        loss = y.sum()
        loss.backward()
        module.zero_grad()
        del x, out, y, loss

    torch.cuda.synchronize()
    baseline_mem = get_cuda_mem()

    alloc_samples = []
    reserved_samples = []
    checkpoint_stats = {}

    step = 100 if num_iters >= 100 else 10

    t0 = time.time()
    for it in range(1, num_iters + 1):
        x = torch.randn(batch_size, seq_len, dim, device=device, requires_grad=True)
        out = module(x)
        y = out[0] if isinstance(out, (tuple, list)) else out
        loss = y.sum()
        loss.backward()
        module.zero_grad()

        del x, out, y, loss

        if it % step == 0 or it == num_iters:
            torch.cuda.synchronize()
            mem = get_cuda_mem()
            checkpoint_stats[it] = mem
            alloc_samples.append(mem["allocated_mb"])
            reserved_samples.append(mem["reserved_mb"])

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    final_mem = get_cuda_mem()

    first_key = min(checkpoint_stats.keys())
    last_key = max(checkpoint_stats.keys())

    alloc_drift = checkpoint_stats[last_key]["allocated_mb"] - checkpoint_stats[first_key]["allocated_mb"]
    reserved_drift = checkpoint_stats[last_key]["reserved_mb"] - checkpoint_stats[first_key]["reserved_mb"]

    frag_ratio = 1.0 - (final_mem["allocated_mb"] / (final_mem["reserved_mb"] + 1e-6))

    result = {
        "module": name,
        "iterations": num_iters,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "dim": dim,
        "elapsed_sec": round(elapsed, 3),
        "ms_per_iter": round((elapsed / num_iters) * 1000, 3),
        "baseline_allocated_mb": round(baseline_mem["allocated_mb"], 2),
        "final_allocated_mb": round(final_mem["allocated_mb"], 2),
        "peak_allocated_mb": round(final_mem["max_allocated_mb"], 2),
        "final_reserved_mb": round(final_mem["reserved_mb"], 2),
        "peak_reserved_mb": round(final_mem["max_reserved_mb"], 2),
        "alloc_drift_mb": round(alloc_drift, 3),
        "reserved_drift_mb": round(reserved_drift, 3),
        "fragmentation_ratio": round(frag_ratio, 4),
        "checkpoints": checkpoint_stats,
    }

    del module
    gc.collect()
    torch.cuda.empty_cache()
    return result


def run_multistream_test():
    """Execute run multistream test routine."""
    torch.cuda.empty_cache()
    device = torch.device("cuda:0")

    stream_attn = torch.cuda.Stream(device=device)
    stream_gemm = torch.cuda.Stream(device=device)

    B, S, D = 4, 512, 768
    x_attn = torch.randn(B, S, D, device=device, requires_grad=True)
    x_gemm = torch.randn(B * S, D, device=device)
    w_u = torch.randn(D, 64, device=device)
    w_v = torch.randn(64, D, device=device)

    gla = GLAttention(hidden_size=D, num_heads=12, head_dim=64).to(device)

    # Warmup
    for _ in range(10):
        with torch.cuda.stream(stream_attn):
            _out_attn = gla(x_attn)[0]
        with torch.cuda.stream(stream_gemm):
            _out_gemm = torch.mm(torch.mm(x_gemm, w_u), w_v)
        torch.cuda.synchronize()

    torch.cuda.synchronize()

    num_rounds = 200

    # Sequential
    torch.cuda.synchronize()
    t_seq_0 = time.time()
    for _ in range(num_rounds):
        ref_attn = gla(x_attn)[0]
        ref_gemm = torch.mm(torch.mm(x_gemm, w_u), w_v)
    torch.cuda.synchronize()
    t_seq = time.time() - t_seq_0

    # Multi-stream concurrent execution
    torch.cuda.synchronize()
    t_par_0 = time.time()
    for _ in range(num_rounds):
        with torch.cuda.stream(stream_attn):
            par_attn = gla(x_attn)[0]
        with torch.cuda.stream(stream_gemm):
            par_gemm = torch.mm(torch.mm(x_gemm, w_u), w_v)
        torch.cuda.synchronize()
    t_par = time.time() - t_par_0

    max_diff_attn = torch.max(torch.abs(ref_attn - par_attn)).item()
    max_diff_gemm = torch.max(torch.abs(ref_gemm - par_gemm)).item()

    speedup = t_seq / max(t_par, 1e-6)

    return {
        "num_rounds": num_rounds,
        "seq_time_sec": round(t_seq, 4),
        "par_time_sec": round(t_par, 4),
        "stream_speedup": round(speedup, 3),
        "max_diff_attn": max_diff_attn,
        "max_diff_gemm": max_diff_gemm,
        "concurrency_clean": bool(max_diff_attn == 0.0 and max_diff_gemm == 0.0),
    }


def main():
    """Execute main routine."""
    print("=== STARTING VRAM AND MEMORY CONCURRENCY PROFILING ===")
    device_name = torch.cuda.get_device_name(0)
    total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {device_name} ({total_vram_gb:.2f} GB)")

    test_modules = [
        ("GLAttention (Linear)", lambda: GLAttention(hidden_size=768, num_heads=12, head_dim=64), 1000),
        (
            "CosformerAttention (Linear)",
            lambda: CosformerAttention(
                config=DummyConfig(), comp_config={"d_model": 768, "num_heads": 12, "head_dim": 64}
            ),
            1000,
        ),
        (
            "GatedDeltaNet (Recurrent/Delta)",
            lambda: GatedDeltaNet(config=DummyConfig(), d_model=768, num_heads=12),
            1000,
        ),
        ("SVDLinearAttention (SVD)", lambda: SVDLinearAttention(config=DummyConfig(), rank=32), 1000),
        (
            "TritonCosformerAttention (Triton)",
            lambda: TritonCosformerAttention(
                config=DummyConfig(), comp_config={"d_model": 768, "num_heads": 12, "head_dim": 64}
            ),
            1000,
        ),
        ("TritonSVDLinearAttention (Triton)", lambda: TritonSVDLinearAttention(config=DummyConfig(), rank=32), 1000),
        ("MambaBlock (SSM Recurrent)", lambda: MambaBlock(config=DummyConfig(), d_model=768), 100),
    ]

    leak_results = []
    for name, fn, iters in test_modules:
        print(f"\n--- Testing {name} over {iters} iterations ---")
        res = run_stress_test(name, fn, num_iters=iters)
        print(f"  Finished in {res['elapsed_sec']}s ({res['ms_per_iter']} ms/iter)")
        print(f"  Peak Alloc: {res['peak_allocated_mb']} MB | Peak Reserved: {res['peak_reserved_mb']} MB")
        print(
            f"  Alloc Drift ({min(res['checkpoints'].keys())} -> {iters}): {res['alloc_drift_mb']} MB | Reserved Drift: {res['reserved_drift_mb']} MB"
        )
        print(f"  Fragmentation Ratio: {res['fragmentation_ratio']:.4f}")
        leak_results.append(res)

    print("\n--- Testing Multi-Stream Concurrency ---")
    stream_res = run_multistream_test()
    print(
        f"  Sequential: {stream_res['seq_time_sec']}s vs Parallel: {stream_res['par_time_sec']}s (Speedup: {stream_res['stream_speedup']}x)"
    )
    print(f"  Max Diff Attn: {stream_res['max_diff_attn']}, Max Diff GEMM: {stream_res['max_diff_gemm']}")
    print(f"  Concurrency clean: {stream_res['concurrency_clean']}")

    full_profile = {
        "device": device_name,
        "total_vram_gb": round(total_vram_gb, 2),
        "leak_tests": leak_results,
        "multistream_test": stream_res,
    }

    os.makedirs("/mnt/d/SVD_linear_Attention/results/profiling", exist_ok=True)
    out_path = "/mnt/d/SVD_linear_Attention/results/profiling/vram_concurrency_profile.json"
    with open(out_path, "w") as f:
        json.dump(full_profile, f, indent=2)
    print(f"\nSaved profile results to {out_path}")


if __name__ == "__main__":
    main()
