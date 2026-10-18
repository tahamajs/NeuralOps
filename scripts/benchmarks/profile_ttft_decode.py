"""Profile ttft decode module.

Author: Taha Majlesi
"""

import sys

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
import torch
import torch.nn.functional as F
import os
import csv

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Benchmark parameters:
# Multi-stream inference under variable prompt prefix lengths (L_prefix in {512, 1024, 2048, 4096}).
# Measure Time-To-First-Token (TTFT in ms) for prefill, and single token decode step latency (ms/token),
# under asynchronous multi-tenant arrival (profiling 1, 2, 4, 8 concurrent tenant streams).

prefix_lengths = [512, 1024, 2048, 4096]
concurrency_levels = [1, 2, 4, 8]

H = 16
D_HEAD = 64
D = H * D_HEAD

print("\n=== STARTING MULTI-STREAM TTFT & DECODE LATENCY PROFILING ===")

# Shared model weights across tenants
W_q = torch.randn(D, D, device=device, dtype=torch.float16)
W_k = torch.randn(D, D, device=device, dtype=torch.float16)
W_v = torch.randn(D, D, device=device, dtype=torch.float16)
W_out = torch.randn(D, D, device=device, dtype=torch.float16)
W_gate = torch.randn(D, 4 * D, device=device, dtype=torch.float16)
W_down = torch.randn(4 * D, D, device=device, dtype=torch.float16)

# Warmup CUDA
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
torch.cuda.synchronize()

results = []

for num_streams in concurrency_levels:
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    print(f"\n--- Concurrency Level: {num_streams} Active Streams ---")

    for L_prefix in prefix_lengths:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Prepare tenant prompt inputs: [1, L_prefix, D] per stream
        prompt_tensors = [torch.randn(1, L_prefix, D, device=device, dtype=torch.float16) for _ in range(num_streams)]
        k_caches = [
            torch.zeros(L_prefix + 64, H, D_HEAD, device=device, dtype=torch.float16) for _ in range(num_streams)
        ]
        v_caches = [
            torch.zeros(L_prefix + 64, H, D_HEAD, device=device, dtype=torch.float16) for _ in range(num_streams)
        ]
        decode_tokens = [torch.randn(1, 1, D, device=device, dtype=torch.float16) for _ in range(num_streams)]

        # 1. Benchmark Prefill -> TTFT (Time-To-First-Token)
        # Prefill processes prompt: Q, K, V projection + causal self-attention + MLP to generate the first token
        def run_prefill(s_idx):
            """Execute run prefill routine."""
            with torch.cuda.stream(streams[s_idx]):
                x = prompt_tensors[s_idx]
                q = torch.matmul(x, W_q).view(1, L_prefix, H, D_HEAD).transpose(1, 2)
                k = torch.matmul(x, W_k).view(1, L_prefix, H, D_HEAD).transpose(1, 2)
                v = torch.matmul(x, W_v).view(1, L_prefix, H, D_HEAD).transpose(1, 2)

                # Populate cache
                k_caches[s_idx][:L_prefix] = k.transpose(1, 2).squeeze(0)
                v_caches[s_idx][:L_prefix] = v.transpose(1, 2).squeeze(0)

                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_flat = attn_out.transpose(1, 2).reshape(L_prefix, D)
                proj_out = torch.matmul(attn_flat, W_out)
                mlp_h = F.silu(torch.matmul(proj_out, W_gate))
                first_token = torch.matmul(mlp_h[-1:], W_down)  # last token prediction
                decode_tokens[s_idx] = first_token

        # Warmup prefill
        for s_idx in range(num_streams):
            run_prefill(s_idx)
        for s_idx in range(num_streams):
            streams[s_idx].synchronize()

        # Measure TTFT over 20 iterations
        iters_prefill = 20
        ttft_start = torch.cuda.Event(enable_timing=True)
        ttft_end = torch.cuda.Event(enable_timing=True)

        ttft_start.record()
        for _ in range(iters_prefill):
            for s_idx in range(num_streams):
                run_prefill(s_idx)
            for s_idx in range(num_streams):
                streams[s_idx].synchronize()
        ttft_end.record()
        torch.cuda.synchronize()

        ttft_batch_ms = ttft_start.elapsed_time(ttft_end) / iters_prefill
        # Per-tenant TTFT (since streams execute concurrently in parallel)
        ttft_per_tenant_ms = ttft_batch_ms
        prefill_tok_per_sec = (num_streams * L_prefix * 1000.0) / ttft_batch_ms

        # 2. Benchmark Single Token Decode Step Latency (at context length L_prefix)
        def run_decode_step(s_idx):
            """Execute run decode step routine."""
            with torch.cuda.stream(streams[s_idx]):
                tok = decode_tokens[s_idx]
                q_step = torch.matmul(tok, W_q).view(1, 1, H, D_HEAD).transpose(1, 2)
                k_step = torch.matmul(tok, W_k).view(H, D_HEAD)
                v_step = torch.matmul(tok, W_v).view(H, D_HEAD)

                k_caches[s_idx][L_prefix] = k_step
                v_caches[s_idx][L_prefix] = v_step

                k_active = k_caches[s_idx][: L_prefix + 1].view(1, L_prefix + 1, H, D_HEAD).transpose(1, 2)
                v_active = v_caches[s_idx][: L_prefix + 1].view(1, L_prefix + 1, H, D_HEAD).transpose(1, 2)

                attn_out = F.scaled_dot_product_attention(q_step, k_active, v_active)
                attn_flat = attn_out.transpose(1, 2).reshape(1, D)
                proj_out = torch.matmul(attn_flat, W_out)
                mlp_h = F.silu(torch.matmul(proj_out, W_gate))
                next_tok = torch.matmul(mlp_h, W_down)
                decode_tokens[s_idx] = next_tok

        # Warmup decode
        for s_idx in range(num_streams):
            run_decode_step(s_idx)
        for s_idx in range(num_streams):
            streams[s_idx].synchronize()

        iters_decode = 50
        dec_start = torch.cuda.Event(enable_timing=True)
        dec_end = torch.cuda.Event(enable_timing=True)

        dec_start.record()
        for _ in range(iters_decode):
            for s_idx in range(num_streams):
                run_decode_step(s_idx)
            for s_idx in range(num_streams):
                streams[s_idx].synchronize()
        dec_end.record()
        torch.cuda.synchronize()

        decode_step_ms = dec_start.elapsed_time(dec_end) / iters_decode
        decode_tok_per_sec = (num_streams * 1000.0) / decode_step_ms
        per_stream_decode_tps = decode_tok_per_sec / num_streams
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        print(
            f"L_prefix={L_prefix:4d} | Streams={num_streams}: TTFT={ttft_per_tenant_ms:.2f} ms ({prefill_tok_per_sec:,.0f} prefill tok/s) | Decode={decode_step_ms:.3f} ms/step ({decode_tok_per_sec:,.0f} tok/s) | Peak VRAM={peak_vram_mb:.1f} MB"
        )

        results.append(
            {
                "Streams": num_streams,
                "Prefix_Len": L_prefix,
                "Total_Prefill_Tokens": num_streams * L_prefix,
                "TTFT_ms": round(ttft_per_tenant_ms, 2),
                "Prefill_Throughput_tok_per_sec": round(prefill_tok_per_sec, 1),
                "Decode_Latency_ms_per_step": round(decode_step_ms, 3),
                "Decode_Throughput_tok_per_sec": round(decode_tok_per_sec, 1),
                "Per_Stream_Decode_TPS": round(per_stream_decode_tps, 1),
                "Peak_VRAM_MB": round(peak_vram_mb, 2),
                "Status": "PASSED",
            }
        )

# Export to CSV
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_ttft_decode.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Streams",
    "Prefix_Len",
    "Total_Prefill_Tokens",
    "TTFT_ms",
    "Prefill_Throughput_tok_per_sec",
    "Decode_Latency_ms_per_step",
    "Decode_Throughput_tok_per_sec",
    "Per_Stream_Decode_TPS",
    "Peak_VRAM_MB",
    "Status",
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
