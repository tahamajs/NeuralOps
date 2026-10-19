# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Profile tail latency jitter module.
"""

import sys

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
import torch
import torch.nn.functional as F
import numpy as np
import os
import csv

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")


num_requests = 100
stream_counts = [1, 2, 4, 8]
batch_per_stream = 1
seq_len = 2048
num_heads = 16
head_dim = 64
dim = num_heads * head_dim

print("\n=== STARTING MULTI-STREAM INFERENCE LATENCY JITTER PROFILING ===")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
torch.cuda.synchronize()

results = []
raw_records = []

for num_streams in stream_counts:
    print(f"\n--- Profiling Concurrency: {num_streams} Active Streams ({num_requests} consecutive requests) ---")
    streams = [torch.cuda.Stream() for _ in range(num_streams)]

    q_tensors = [
        torch.randn(batch_per_stream, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_streams)
    ]
    k_tensors = [
        torch.randn(batch_per_stream, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_streams)
    ]
    v_tensors = [
        torch.randn(batch_per_stream, num_heads, seq_len, head_dim, device=device, dtype=torch.float16)
        for _ in range(num_streams)
    ]

    W_proj = torch.randn(dim, dim, device=device, dtype=torch.float16)

    for i in range(num_streams):
        with torch.cuda.stream(streams[i]):
            out = F.scaled_dot_product_attention(q_tensors[i], k_tensors[i], v_tensors[i])
            out_flat = out.transpose(1, 2).reshape(batch_per_stream * seq_len, dim)
            _ = torch.matmul(out_flat, W_proj)
    torch.cuda.synchronize()

    latencies_ms = []

    torch.cuda.reset_peak_memory_stats()

    start_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_requests)]
    end_events = [torch.cuda.Event(enable_timing=True) for _ in range(num_requests)]

    for r in range(num_requests):
        start_events[r].record()
        for s_idx in range(num_streams):
            with torch.cuda.stream(streams[s_idx]):
                out = F.scaled_dot_product_attention(q_tensors[s_idx], k_tensors[s_idx], v_tensors[s_idx])
                out_flat = out.transpose(1, 2).reshape(batch_per_stream * seq_len, dim)
                _ = torch.matmul(out_flat, W_proj)

        for s_idx in range(num_streams):
            streams[s_idx].synchronize()
        end_events[r].record()

    torch.cuda.synchronize()

    for r in range(num_requests):
        lat = start_events[r].elapsed_time(end_events[r])
        latencies_ms.append(lat)

    latencies_arr = np.array(latencies_ms)
    mean_lat = float(np.mean(latencies_arr))
    std_lat = float(np.std(latencies_arr))
    p50_lat = float(np.percentile(latencies_arr, 50))
    p90_lat = float(np.percentile(latencies_arr, 90))
    p95_lat = float(np.percentile(latencies_arr, 95))
    p99_lat = float(np.percentile(latencies_arr, 99))
    max_lat = float(np.max(latencies_arr))
    min_lat = float(np.min(latencies_arr))

    jitter_ratio = (p99_lat - p50_lat) / p50_lat * 100.0 if p50_lat > 0 else 0.0

    total_time_sec = float(np.sum(latencies_arr)) / 1000.0
    total_requests_served = num_requests * num_streams
    qps = total_requests_served / total_time_sec
    tokens_per_sec = (total_requests_served * seq_len) / total_time_sec
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    print(f"Results for {num_streams} streams:")
    print(f"  P50: {p50_lat:.3f} ms | P90: {p90_lat:.3f} ms | P99: {p99_lat:.3f} ms | Max: {max_lat:.3f} ms")
    print(f"  Jitter (P99-P50)/P50: {jitter_ratio:.1f}% | QPS: {qps:,.1f} req/s | TPS: {tokens_per_sec:,.0f} tok/s")
    print(f"  Peak VRAM: {peak_vram_mb:.2f} MB")

    results.append(
        {
            "Streams": num_streams,
            "Total_Requests": total_requests_served,
            "Seq_Len": seq_len,
            "Mean_Latency_ms": round(mean_lat, 3),
            "Std_Dev_ms": round(std_lat, 3),
            "Min_Latency_ms": round(min_lat, 3),
            "P50_ms": round(p50_lat, 3),
            "P90_ms": round(p90_lat, 3),
            "P95_ms": round(p95_lat, 3),
            "P99_ms": round(p99_lat, 3),
            "Max_Latency_ms": round(max_lat, 3),
            "Jitter_Ratio_pct": round(jitter_ratio, 2),
            "QPS": round(qps, 1),
            "Tokens_Per_Sec": round(tokens_per_sec, 0),
            "Peak_VRAM_MB": round(peak_vram_mb, 2),
        }
    )

csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_tail_latency_jitter.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Streams",
    "Total_Requests",
    "Seq_Len",
    "Mean_Latency_ms",
    "Std_Dev_ms",
    "Min_Latency_ms",
    "P50_ms",
    "P90_ms",
    "P95_ms",
    "P99_ms",
    "Max_Latency_ms",
    "Jitter_Ratio_pct",
    "QPS",
    "Tokens_Per_Sec",
    "Peak_VRAM_MB",
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
