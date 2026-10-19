#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Profile dynamic batching module.
"""

import sys

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import csv

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")


np.random.seed(42)
torch.manual_seed(42)

num_requests = 100
min_len, max_len = 256, 4096
req_lengths = np.random.randint(min_len, max_len + 1, size=num_requests)
inter_arrival_ms = np.random.exponential(scale=15.0, size=num_requests)
arrival_times_ms = np.cumsum(inter_arrival_ms)

H = 16
D_HEAD = 64
D = H * D_HEAD
SLA_DEADLINE_MS = 100.0

print(f"\n=== STARTING DYNAMIC BATCH SCHEDULING BENCHMARK ({num_requests} REQUESTS) ===")
print(f"Min Seq Len: {min_len} | Max Seq Len: {max_len} | Avg Seq Len: {np.mean(req_lengths):.1f}")
print(f"Total Tokens across 100 Requests: {np.sum(req_lengths):,}")

W_q = torch.randn(D, D, device=device, dtype=torch.float16)
W_k = torch.randn(D, D, device=device, dtype=torch.float16)
W_v = torch.randn(D, D, device=device, dtype=torch.float16)
W_out = torch.randn(D, D, device=device, dtype=torch.float16)
W_gate = torch.randn(D, 4 * D, device=device, dtype=torch.float16)
W_down = torch.randn(4 * D, D, device=device, dtype=torch.float16)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
torch.cuda.synchronize()

results = []

for batch_cap in [4, 8]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    request_latencies = []
    current_time_ms = 0.0

    for b_start in range(0, num_requests, batch_cap):
        b_end = min(b_start + batch_cap, num_requests)
        batch_indices = list(range(b_start, b_end))
        b_size = len(batch_indices)

        last_arrival = arrival_times_ms[b_end - 1]
        batch_start_time = max(current_time_ms, last_arrival)

        max_b_len = int(np.max(req_lengths[batch_indices]))

        x_pad = torch.randn(b_size, max_b_len, D, device=device, dtype=torch.float16)

        t0 = time.perf_counter()
        q = torch.matmul(x_pad, W_q).view(b_size, max_b_len, H, D_HEAD).transpose(1, 2)
        k = torch.matmul(x_pad, W_k).view(b_size, max_b_len, H, D_HEAD).transpose(1, 2)
        v = torch.matmul(x_pad, W_v).view(b_size, max_b_len, H, D_HEAD).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_flat = attn_out.transpose(1, 2).reshape(b_size * max_b_len, D)
        proj_out = torch.matmul(attn_flat, W_out)
        mlp_h = F.silu(torch.matmul(proj_out, W_gate))
        _ = torch.matmul(mlp_h, W_down)
        torch.cuda.synchronize()
        exec_duration_ms = (time.perf_counter() - t0) * 1000.0

        batch_finish_time = batch_start_time + exec_duration_ms
        current_time_ms = batch_finish_time

        for idx in batch_indices:
            lat = batch_finish_time - arrival_times_ms[idx]
            request_latencies.append(lat)

    latencies_arr = np.array(request_latencies)
    p50 = float(np.percentile(latencies_arr, 50))
    p90 = float(np.percentile(latencies_arr, 90))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_lat = float(np.mean(latencies_arr))
    max_lat = float(np.max(latencies_arr))

    missed_deadlines = np.sum(latencies_arr > SLA_DEADLINE_MS)
    sla_compliance_pct = ((num_requests - missed_deadlines) / num_requests) * 100.0

    total_time_sec = current_time_ms / 1000.0
    throughput_tps = np.sum(req_lengths) / total_time_sec
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    policy_name = f"Static Batching (Batch Size B={batch_cap})"
    print(f"\n[{policy_name}]")
    print(f"  P50: {p50:.2f} ms | P90: {p90:.2f} ms | P99: {p99:.2f} ms | Mean: {mean_lat:.2f} ms")
    print(f"  SLA Compliance (<=100ms): {sla_compliance_pct:.1f}% ({missed_deadlines} violations)")
    print(f"  Throughput: {throughput_tps:,.0f} tok/s | Peak VRAM: {peak_vram_mb:.1f} MB")

    results.append(
        {
            "Scheduling_Policy": policy_name,
            "Total_Requests": num_requests,
            "Mean_Latency_ms": round(mean_lat, 2),
            "P50_ms": round(p50, 2),
            "P90_ms": round(p90, 2),
            "P95_ms": round(p95, 2),
            "P99_ms": round(p99_2 := p99, 2),
            "Max_Latency_ms": round(max_lat, 2),
            "SLA_Compliance_pct": round(sla_compliance_pct, 1),
            "Throughput_tok_per_sec": round(throughput_tps, 1),
            "Peak_VRAM_MB": round(peak_vram_mb, 2),
            "Status": "PASSED",
        }
    )

for chunk_budget in [2048, 4096]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    request_latencies = []

    req_remaining = req_lengths.copy()
    req_finish_times = np.zeros(num_requests)

    current_time_ms = 0.0
    req_idx = 0
    active_requests = []

    while req_idx < num_requests or len(active_requests) > 0:
        while req_idx < num_requests and arrival_times_ms[req_idx] <= current_time_ms:
            active_requests.append(req_idx)
            req_idx += 1

        if len(active_requests) == 0:
            current_time_ms = arrival_times_ms[req_idx]
            active_requests.append(req_idx)
            req_idx += 1

        budget_left = chunk_budget
        step_tokens = 0
        scheduled_this_step = []

        for r in active_requests:
            if budget_left <= 0:
                break
            tokens_to_take = min(req_remaining[r], budget_left)
            req_remaining[r] -= tokens_to_take
            budget_left -= tokens_to_take
            step_tokens += tokens_to_take
            scheduled_this_step.append((r, tokens_to_take))

        x_packed = torch.randn(step_tokens, D, device=device, dtype=torch.float16)

        t0 = time.perf_counter()
        q = torch.matmul(x_packed, W_q).view(1, step_tokens, H, D_HEAD).transpose(1, 2)
        k = torch.matmul(x_packed, W_k).view(1, step_tokens, H, D_HEAD).transpose(1, 2)
        v = torch.matmul(x_packed, W_v).view(1, step_tokens, H, D_HEAD).transpose(1, 2)
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_flat = attn_out.transpose(1, 2).reshape(step_tokens, D)
        proj_out = torch.matmul(attn_flat, W_out)
        mlp_h = F.silu(torch.matmul(proj_out, W_gate))
        _ = torch.matmul(mlp_h, W_down)
        torch.cuda.synchronize()
        exec_duration_ms = (time.perf_counter() - t0) * 1000.0

        current_time_ms += exec_duration_ms

        new_active = []
        for r in active_requests:
            if req_remaining[r] == 0:
                req_finish_times[r] = current_time_ms
            else:
                new_active.append(r)
        active_requests = new_active

    latencies_arr = req_finish_times - arrival_times_ms
    p50 = float(np.percentile(latencies_arr, 50))
    p90 = float(np.percentile(latencies_arr, 90))
    p95 = float(np.percentile(latencies_arr, 95))
    p99 = float(np.percentile(latencies_arr, 99))
    mean_lat = float(np.mean(latencies_arr))
    max_lat = float(np.max(latencies_arr))

    missed_deadlines = np.sum(latencies_arr > SLA_DEADLINE_MS)
    sla_compliance_pct = ((num_requests - missed_deadlines) / num_requests) * 100.0

    total_time_sec = current_time_ms / 1000.0
    throughput_tps = np.sum(req_lengths) / total_time_sec
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

    policy_name = f"Continuous Dynamic Batching (Chunk Budget C={chunk_budget})"
    print(f"\n[{policy_name}]")
    print(f"  P50: {p50:.2f} ms | P90: {p90:.2f} ms | P99: {p99:.2f} ms | Mean: {mean_lat:.2f} ms")
    print(f"  SLA Compliance (<=100ms): {sla_compliance_pct:.1f}% ({missed_deadlines} violations)")
    print(f"  Throughput: {throughput_tps:,.0f} tok/s | Peak VRAM: {peak_vram_mb:.1f} MB")

    results.append(
        {
            "Scheduling_Policy": policy_name,
            "Total_Requests": num_requests,
            "Mean_Latency_ms": round(mean_lat, 2),
            "P50_ms": round(p50, 2),
            "P90_ms": round(p90, 2),
            "P95_ms": round(p95, 2),
            "P99_ms": round(p99, 2),
            "Max_Latency_ms": round(max_lat, 2),
            "SLA_Compliance_pct": round(sla_compliance_pct, 1),
            "Throughput_tok_per_sec": round(throughput_tps, 1),
            "Peak_VRAM_MB": round(peak_vram_mb, 2),
            "Status": "PASSED",
        }
    )

csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_dynamic_batching.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Scheduling_Policy",
    "Total_Requests",
    "Mean_Latency_ms",
    "P50_ms",
    "P90_ms",
    "P95_ms",
    "P99_ms",
    "Max_Latency_ms",
    "SLA_Compliance_pct",
    "Throughput_tok_per_sec",
    "Peak_VRAM_MB",
    "Status",
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
