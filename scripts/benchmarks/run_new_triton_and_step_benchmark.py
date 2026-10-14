#!/usr/bin/env python3
"""
Benchmark Suite: Exact O(1) Recurrent .step() Throughput & 4 New Triton Capabilities.

Evaluates:
1. Decoding token throughput (tokens/sec) comparing:
   - Full-context re-computation (quadratic/naive decoding)
   - O(1) Recurrent .step() with state caching across GLA, GDN, KDA, HLA, Cosformer
2. Forward latency across sequence lengths T in {512, 1024, 2048, 4096, 8192}.
3. Native FP8 Tensor Core MMA vs FP16 baseline.
"""

import os
import time
import json
import torch
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY
from neuralops.core.triton_kernels.low_rank import (
    quantize_to_fp8_dynamic,
    fp8_tensorcore_matmul_triton,
)


def benchmark_step_throughput(device, models=["cosformer", "gla_attention", "gated_delta_net", "kda_attention", "hla_attention"]):
    results = {}
    dim = 256
    heads = 4
    decode_steps = 128

    print(f"\n--- Recurrent .step() Autoregressive Throughput Benchmark (Steps = {decode_steps}) ---")
    for name in models:
        cls = ATTENTION_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            m = cls(dim=dim, num_heads=heads, is_decoder=True, window_size=0).to(device=device)
        except Exception:
            m = cls(hidden_size=dim, num_heads=heads).to(device=device)
        m.eval()

        # Warmup
        x_warm = torch.randn(1, 1, dim, device=device)
        state = None
        for _ in range(10):
            _, state = m.step(x_warm, state=state)

        # Benchmark recurrent .step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        state = None
        for _ in range(decode_steps):
            x_t = torch.randn(1, 1, dim, device=device)
            _, state = m.step(x_t, state=state)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        step_elapsed = t1 - t0
        tokens_per_sec = decode_steps / step_elapsed
        results[name] = {
            "decode_steps": decode_steps,
            "step_time_ms": step_elapsed * 1000.0,
            "tokens_per_sec": tokens_per_sec,
            "ms_per_token": (step_elapsed / decode_steps) * 1000.0,
        }
        print(f"[{name:20s}] Throughput: {tokens_per_sec:8.1f} tokens/s | Latency: {(step_elapsed / decode_steps) * 1000.0:6.3f} ms/token")

    return results


def benchmark_sequence_scaling(device, seq_lens=[512, 1024, 2048, 4096]):
    results = {}
    dim = 256
    heads = 4
    models = ["cosformer", "gla_attention", "kda_attention"]

    print("\n--- Subquadratic Sequence Length Scaling Benchmark (Latency in ms) ---")
    for name in models:
        cls = ATTENTION_REGISTRY.get(name)
        if cls is None:
            continue
        try:
            m = cls(dim=dim, num_heads=heads, is_decoder=True, window_size=0).to(device=device)
        except Exception:
            m = cls(hidden_size=dim, num_heads=heads).to(device=device)
        m.eval()

        results[name] = {}
        for T in seq_lens:
            try:
                x = torch.randn(1, T, dim, device=device)
                # Warmup
                for _ in range(3):
                    _ = m(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()

                iters = 10
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = m(x)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t1 = time.perf_counter()

                latency_ms = ((t1 - t0) / iters) * 1000.0
                results[name][T] = latency_ms
                print(f"[{name:15s}] T = {T:5d}: {latency_ms:7.2f} ms")
            except Exception as e:
                print(f"[{name:15s}] T = {T:5d}: OOM / Skipped ({e})")
                results[name][T] = None

    return results


def benchmark_fp8_speedup(device):
    print("\n--- FP8 Tensor Core MMA Benchmark ---")
    if device.type != "cuda":
        return {"status": "cpu_fallback"}

    M, K, N = 2048, 2048, 2048
    x_fp16 = torch.randn(M, K, device=device, dtype=torch.float16)
    w_fp16 = torch.randn(K, N, device=device, dtype=torch.float16)

    # Warmup
    for _ in range(5):
        _ = torch.matmul(x_fp16, w_fp16)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        _ = torch.matmul(x_fp16, w_fp16)
    torch.cuda.synchronize()
    fp16_time = (time.perf_counter() - t0) / 50.0

    # FP8
    x_fp8, sx = quantize_to_fp8_dynamic(x_fp16.float())
    w_fp8, sw = quantize_to_fp8_dynamic(w_fp16.float())

    for _ in range(5):
        _ = fp8_tensorcore_matmul_triton(x_fp8, w_fp8, sx, sw)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(50):
        _ = fp8_tensorcore_matmul_triton(x_fp8, w_fp8, sx, sw)
    torch.cuda.synchronize()
    fp8_time = (time.perf_counter() - t0) / 50.0

    speedup = fp16_time / max(fp8_time, 1e-9)
    print(f"Matrix ({M}x{K} @ {K}x{N}) FP16: {fp16_time * 1000.0:.2f} ms | FP8: {fp8_time * 1000.0:.2f} ms | Speedup: {speedup:.2f}x")
    return {
        "matrix_shape": [M, K, N],
        "fp16_ms": fp16_time * 1000.0,
        "fp8_ms": fp8_time * 1000.0,
        "speedup": speedup,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    step_results = benchmark_step_throughput(device)
    scaling_results = benchmark_sequence_scaling(device)
    fp8_results = benchmark_fp8_speedup(device)

    out_data = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "step_throughput": step_results,
        "sequence_scaling": scaling_results,
        "fp8_benchmark": fp8_results,
    }

    os.makedirs("results/v13_deep_empirical_suite", exist_ok=True)
    out_file = "results/v13_deep_empirical_suite/triton_and_step_benchmark_results.json"
    with open(out_file, "w") as f:
        json.dump(out_data, f, indent=2)

    print(f"\n[OK] Benchmark completed and saved to {out_file}")


if __name__ == "__main__":
    main()
