# -*- coding: utf-8 -*-
"""Memory Bandwidth, Cache Saturation, and Latency Benchmark for Core CUDA Kernels.

Evaluates memory bandwidth, L2-cache residency, and DRAM saturation across
realistic LLM training configurations on NVIDIA GeForce RTX 4070 / Ada Lovelace.
"""

import argparse
import csv
import json
import math
import os
import sys
import torch
import torch.nn.functional as F

DEFAULT_PEAK_BW_GBPS = 256.0
DEFAULT_L2_CACHE_BYTES = 32 * 1024 * 1024  # 32 MB RTX 4070 Ada Lovelace

DEFAULT_CONFIGS = [
    (4, 512, 2048),
    (8, 512, 2048),
    (4, 1024, 2048),
    (8, 1024, 2048),
    (4, 512, 4096),
    (8, 512, 4096),
    (16, 512, 2048),
    (4, 1024, 4096),
]

QUICK_CONFIGS = [
    (4, 512, 2048),
    (4, 1024, 2048),
]


def flush_l2_cache(device="cuda", size_bytes=32 * 1024 * 1024 * 2):
    """Write a buffer larger than L2 to evict all cached data."""
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return
    flush_buf = torch.empty(size_bytes, dtype=torch.uint8, device=device)
    flush_buf.zero_()
    torch.cuda.synchronize()
    del flush_buf


def compute_bandwidth_stats(
    bytes_transferred: int,
    time_ms: float,
    L2_cache_size_bytes: int = DEFAULT_L2_CACHE_BYTES,
    DRAM_peak_bw_bytes_per_sec: float = DEFAULT_PEAK_BW_GBPS * 1e9,
):
    """Compute achieved bandwidth (GB/s) and DRAM saturation percentage."""
    if time_ms <= 0:
        return {
            "achieved_bw_gbps": 0.0,
            "saturation_pct": float("nan"),
            "regime": "N/A",
        }
    achieved_bw = bytes_transferred / (time_ms * 1e-3)  # bytes/sec
    achieved_bw_gbps = achieved_bw / 1e9

    if bytes_transferred < 0.9 * L2_cache_size_bytes:
        # Workload is L2-resident; DRAM saturation is not bottlenecked by HBM
        return {
            "achieved_bw_gbps": achieved_bw_gbps,
            "saturation_pct": float("nan"),
            "regime": "L2-resident",
        }
    else:
        saturation = (achieved_bw / DRAM_peak_bw_bytes_per_sec) * 100.0
        return {
            "achieved_bw_gbps": achieved_bw_gbps,
            "saturation_pct": min(saturation, 100.0),
            "regime": "DRAM-bound",
        }


def format_speedup(cuda_ms: float, pytorch_ms: float) -> str:
    """Format speedup ratio cleanly."""
    if cuda_ms <= 0:
        return "N/A"
    speedup = pytorch_ms / cuda_ms
    if speedup >= 10.0:
        return f"{speedup:.1f}x"
    elif speedup >= 1.0:
        return f"{speedup:.2f}x"
    elif speedup >= 0.01:
        return f"{speedup:.3f}x"
    else:
        return f"{1.0 / speedup:.0f}x SLOWER"


def time_fn(fn, device="cuda", warmup=10, iters=30):
    """Accurately measure CUDA event execution latency with L2 cache flushes."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available() and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        flush_l2_cache(device=device)
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / iters
    else:
        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        t1 = time.perf_counter()
        return ((t1 - t0) / iters) * 1000.0


def run_bandwidth_benchmark(
    configs=None,
    device_str="cuda:0",
    dtype_str="float16",
    warmup=10,
    repeat=30,
    peak_bw_gbps=DEFAULT_PEAK_BW_GBPS,
    l2_cache_mb=32.0,
    output_csv=None,
    output_json=None,
):
    """Run full LLM bandwidth benchmarking sweep."""
    if configs is None:
        configs = DEFAULT_CONFIGS

    device = torch.device(device_str if (torch.cuda.is_available() and "cuda" in device_str) else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)

    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    dtype = dtype_map.get(dtype_str.lower(), torch.float16)
    l2_cache_bytes = int(l2_cache_mb * 1024 * 1024)
    dram_peak_bytes = peak_bw_gbps * 1e9

    # Lazy import of CUDA kernel modules
    from neuralops.core.cuda_kernels.fused_dropout_add.dropout_add_autograd import (
        _get_module as get_fda,
        fused_dropout_add_pytorch_ref,
    )
    from neuralops.core.cuda_kernels.fused_norm_linear.norm_linear_autograd import (
        _get_module as get_fnl,
        fused_norm_linear_pytorch_ref,
    )
    from neuralops.core.cuda_kernels.rms_variance_match.rms_variance_match_autograd import (
        _get_module as get_rvm,
        rms_variance_match_pytorch_ref,
    )
    from neuralops.core.cuda_kernels.aegis_gate_norm.aegis_gate_norm_autograd import (
        _get_module as get_agn,
        aegis_gate_norm_pytorch_ref,
    )

    mod_fda = get_fda()
    mod_fnl = get_fnl()
    mod_rvm = get_rvm()
    mod_agn = get_agn()

    dev_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    print("=" * 140)
    print(f"Realistic LLM Training Batch Configurations Benchmark ({dev_name})")
    print(f"Hardware Peak Memory Bandwidth: {peak_bw_gbps:.1f} GB/s | L2 Cache Size: {l2_cache_mb:.0f} MB | Precision: {dtype_str}")
    print("=" * 140)

    results = []

    for B, S, H in configs:
        N = B * S

        # 1. fused_dropout_add
        x = torch.randn(N, H, device=device, dtype=dtype)
        res = torch.randn(N, H, device=device, dtype=dtype)
        elem_bytes = x.element_size()
        bytes_fda = 3 * N * H * elem_bytes
        seed_val = int(torch.randint(0, 2**31, (1,), device="cpu").item())

        t_fda_c = time_fn(lambda: mod_fda.fused_dropout_add_fwd(x, res, 0.1, False, seed_val), device=device, warmup=warmup, iters=repeat)
        t_fda_p = time_fn(lambda: fused_dropout_add_pytorch_ref(x, res, 0.1, False), device=device, warmup=warmup, iters=repeat)
        stats_fda = compute_bandwidth_stats(bytes_fda, t_fda_c, L2_cache_size_bytes=l2_cache_bytes, DRAM_peak_bw_bytes_per_sec=dram_peak_bytes)

        results.append({
            "kernel": "fused_dropout_add",
            "cfg": f"B={B}, S={S}, H={H}",
            "cuda_ms": t_fda_c,
            "ref_ms": t_fda_p,
            "speedup_str": format_speedup(t_fda_c, t_fda_p),
            "speedup_raw": (t_fda_p / t_fda_c) if t_fda_c > 0 else 1.0,
            "bw_gbps": stats_fda["achieved_bw_gbps"],
            "sat_pct": stats_fda["saturation_pct"],
            "regime": stats_fda["regime"],
        })

        # 2. rms_variance_match
        y = torch.randn(N, H, device=device, dtype=dtype)
        v = torch.randn(N, H, device=device, dtype=dtype)
        bytes_rvm = 3 * N * H * elem_bytes

        t_rvm_c = time_fn(lambda: mod_rvm.rms_variance_match_fwd(y, v, 0.25, 4.0), device=device, warmup=warmup, iters=repeat)
        t_rvm_p = time_fn(lambda: rms_variance_match_pytorch_ref(y, v, 0.25, 4.0), device=device, warmup=warmup, iters=repeat)
        stats_rvm = compute_bandwidth_stats(bytes_rvm, t_rvm_c, L2_cache_size_bytes=l2_cache_bytes, DRAM_peak_bw_bytes_per_sec=dram_peak_bytes)

        results.append({
            "kernel": "rms_variance_match",
            "cfg": f"B={B}, S={S}, H={H}",
            "cuda_ms": t_rvm_c,
            "ref_ms": t_rvm_p,
            "speedup_str": format_speedup(t_rvm_c, t_rvm_p),
            "speedup_raw": (t_rvm_p / t_rvm_c) if t_rvm_c > 0 else 1.0,
            "bw_gbps": stats_rvm["achieved_bw_gbps"],
            "sat_pct": stats_rvm["saturation_pct"],
            "regime": stats_rvm["regime"],
        })

        # 3. aegis_gate_norm
        b0 = torch.randn(N, H, device=device, dtype=dtype)
        b1 = torch.randn(N, H, device=device, dtype=dtype)
        w_gate = torch.randn(N, 2, device=device, dtype=torch.float32)
        bytes_agn = (3 * N * H * elem_bytes) + (N * 2 * 4)

        t_agn_c = time_fn(lambda: mod_agn.aegis_gate_norm_2branch_fwd(b0, b1, w_gate, 1e-5), device=device, warmup=warmup, iters=repeat)
        t_agn_p = time_fn(lambda: aegis_gate_norm_pytorch_ref(b0, b1, w_gate, 1e-5), device=device, warmup=warmup, iters=repeat)
        stats_agn = compute_bandwidth_stats(bytes_agn, t_agn_c, L2_cache_size_bytes=l2_cache_bytes, DRAM_peak_bw_bytes_per_sec=dram_peak_bytes)

        results.append({
            "kernel": "aegis_gate_norm",
            "cfg": f"B={B}, S={S}, H={H}",
            "cuda_ms": t_agn_c,
            "ref_ms": t_agn_p,
            "speedup_str": format_speedup(t_agn_c, t_agn_p),
            "speedup_raw": (t_agn_p / t_agn_c) if t_agn_c > 0 else 1.0,
            "bw_gbps": stats_agn["achieved_bw_gbps"],
            "sat_pct": stats_agn["saturation_pct"],
            "regime": stats_agn["regime"],
        })

        # 4. fused_norm_linear
        w_lin = torch.randn(H, H, device=device, dtype=dtype)
        bytes_fnl = (2 * N * H + H * H) * elem_bytes
        empty_b = torch.empty(0, device=device, dtype=dtype)

        t_fnl_c = time_fn(lambda: mod_fnl.fused_norm_linear_fwd(x, w_lin, empty_b, 1e-6)[0], device=device, warmup=warmup, iters=repeat)
        t_fnl_p = time_fn(lambda: fused_norm_linear_pytorch_ref(x, w_lin, None, 1e-6), device=device, warmup=warmup, iters=repeat)
        stats_fnl = compute_bandwidth_stats(bytes_fnl, t_fnl_c, L2_cache_size_bytes=l2_cache_bytes, DRAM_peak_bw_bytes_per_sec=dram_peak_bytes)

        results.append({
            "kernel": "fused_norm_linear",
            "cfg": f"B={B}, S={S}, H={H}",
            "cuda_ms": t_fnl_c,
            "ref_ms": t_fnl_p,
            "speedup_str": format_speedup(t_fnl_c, t_fnl_p),
            "speedup_raw": (t_fnl_p / t_fnl_c) if t_fnl_c > 0 else 1.0,
            "bw_gbps": stats_fnl["achieved_bw_gbps"],
            "sat_pct": stats_fnl["saturation_pct"],
            "regime": stats_fnl["regime"],
        })

        print(f"Config B={B}, S={S}, H={H} completed.")

    h_k = "Kernel"
    h_cfg = "Config"
    h_c = "CUDA (ms)"
    h_p = "PyTorch (ms)"
    h_s = "Speedup"
    h_bw = "Bandwidth (GB/s)"
    h_sat = "Saturation %"
    h_reg = "Regime"

    print("\n" + "=" * 140)
    print(f"| {h_k:20} | {h_cfg:18} | {h_c:10} | {h_p:12} | {h_s:14} | {h_bw:18} | {h_sat:18} | {h_reg:12} |")
    print("|" + "-" * 22 + "|" + "-" * 20 + "|" + "-" * 12 + "|" + "-" * 14 + "|" + "-" * 16 + "|" + "-" * 20 + "|" + "-" * 20 + "|" + "-" * 14 + "|")

    for r in results:
        k = r["kernel"]
        cfg = r["cfg"]
        c_ms = r["cuda_ms"]
        p_ms = r["ref_ms"]
        spd_str = r["speedup_str"]
        bw = r["bw_gbps"]
        sat = r["sat_pct"]
        reg = r["regime"]
        sat_str = "N/A (L2-resident)" if math.isnan(sat) else f"{sat:10.1f}%"
        print(f"| {k:20} | {cfg:18} | {c_ms:8.4f}ms | {p_ms:10.4f}ms | {spd_str:>14} | {bw:15.2f} GB/s | {sat_str:>18} | {reg:12} |")
    print("=" * 140)

    # Save to CSV
    if output_csv:
        os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Kernel", "Config", "CUDA (ms)", "PyTorch (ms)", "Speedup", "Bandwidth (GB/s)", "Saturation %", "Regime"])
            for r in results:
                sat_val = "N/A (L2-resident)" if math.isnan(r["sat_pct"]) else f"{r['sat_pct']:.1f}%"
                writer.writerow([
                    r["kernel"],
                    r["cfg"],
                    f"{r['cuda_ms']:.4f}",
                    f"{r['ref_ms']:.4f}",
                    r["speedup_str"],
                    f"{r['bw_gbps']:.2f}",
                    sat_val,
                    r["regime"],
                ])
        print(f"\nSaved benchmark CSV results to {output_csv}")

    # Save to JSON
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        json_data = []
        for r in results:
            item = dict(r)
            if math.isnan(item["sat_pct"]):
                item["sat_pct"] = None
            json_data.append(item)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        print(f"Saved benchmark JSON results to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Realistic LLM Training Batch Configurations Benchmark")
    parser.add_argument("--output", type=str, default="results/benchmark_4_kernels_corrected.csv", help="Path to save CSV output")
    parser.add_argument("--output-json", type=str, default=None, help="Path to save JSON output (defaults to same stem as --output)")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device to execute benchmark")
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32", "bfloat16"], help="Data type precision")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup iterations")
    parser.add_argument("--repeat", type=int, default=30, help="Number of measurement repetitions")
    parser.add_argument("--quick", action="store_true", help="Quick verification pass with fewer configs and iterations")
    parser.add_argument("--peak-bw", type=float, default=DEFAULT_PEAK_BW_GBPS, help="Hardware Peak Bandwidth in GB/s")
    parser.add_argument("--l2-cache-mb", type=float, default=32.0, help="Hardware L2 Cache size in MB")
    args = parser.parse_args()

    configs = QUICK_CONFIGS if args.quick else DEFAULT_CONFIGS
    warmup = 2 if args.quick else args.warmup
    repeat = 5 if args.quick else args.repeat
    output_json = args.output_json
    if output_json is None and args.output:
        base, _ = os.path.splitext(args.output)
        output_json = base + ".json"

    run_bandwidth_benchmark(
        configs=configs,
        device_str=args.device,
        dtype_str=args.dtype,
        warmup=warmup,
        repeat=repeat,
        peak_bw_gbps=args.peak_bw,
        l2_cache_mb=args.l2_cache_mb,
        output_csv=args.output,
        output_json=output_json,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
