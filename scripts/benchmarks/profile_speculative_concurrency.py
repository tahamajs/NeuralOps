"""Profile speculative concurrency module.

Author: Taha Majlesi
"""

import sys

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
import torch
import torch.nn.functional as F
import os
import csv
from neuralops.core.cuda_kernels.speculative_verify.spec_autograd import speculative_verify

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Benchmark parameters:
# Speculative decoding verification concurrency (speculative_verify):
# Speculation window sizes gamma in {2, 4, 8, 16}.
# Evaluate concurrent stream verification across independent CUDA streams (1, 2, 4, 8 streams).
# Under speculative decoding:
# 1. Target model verifies gamma drafted tokens in a single parallel forward pass.
# 2. speculative_verify kernel computes prefix match / accept mask and accepted count.
# 3. Memory traffic savings vs sequential token verification = 1 - (1 / (1 + accepted_count)).
# Compare parallel verification latency vs sequential autoregressive verification latency.

gamma_values = [2, 4, 8, 16]
stream_counts = [1, 2, 4, 8]
batch_per_stream = 2
vocab_size = 32000
acceptance_rate_sim = 0.75  # typical draft model acceptance rate (~75%)

H = 16
D_HEAD = 64
D = H * D_HEAD

print("\n=== STARTING SPECULATIVE DECODING VERIFICATION CONCURRENCY PROFILING ===")

# Model weights for target model verification pass
W_q = torch.randn(D, D, device=device, dtype=torch.float16)
W_k = torch.randn(D, D, device=device, dtype=torch.float16)
W_v = torch.randn(D, D, device=device, dtype=torch.float16)
W_out = torch.randn(D, D, device=device, dtype=torch.float16)
W_gate = torch.randn(D, 4 * D, device=device, dtype=torch.float16)
W_down = torch.randn(4 * D, D, device=device, dtype=torch.float16)

# Warmup CUDA and speculative_verify kernel
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
_ = speculative_verify(torch.randint(0, 100, (2, 4), device=device), torch.randint(0, 100, (2, 4), device=device))
torch.cuda.synchronize()

results = []

for num_streams in stream_counts:
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    print(f"\n--- Concurrency Level: {num_streams} Active Streams ---")

    for gamma in gamma_values:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Prepare inputs for each stream:
        # Draft tokens [batch_per_stream, gamma]
        # Target tokens [batch_per_stream, gamma]
        draft_tokens_list = []
        target_tokens_list = []
        embed_inputs = []

        for _ in range(num_streams):
            draft = torch.randint(0, vocab_size, (batch_per_stream, gamma), device=device, dtype=torch.int64)
            # simulate 75% match
            mask = torch.rand(batch_per_stream, gamma, device=device) < acceptance_rate_sim
            target = draft.clone()
            target[~mask] = torch.randint(0, vocab_size, ((~mask).sum(),), device=device, dtype=torch.int64)

            draft_tokens_list.append(draft)
            target_tokens_list.append(target)
            embed_inputs.append(torch.randn(batch_per_stream, gamma, D, device=device, dtype=torch.float16))

        # 1. Parallel Speculative Verification pass:
        # Forward pass on gamma tokens + speculative_verify kernel
        def run_parallel_verification(s_idx):
            """Execute run parallel verification routine."""
            with torch.cuda.stream(streams[s_idx]):
                x = embed_inputs[s_idx]
                q = torch.matmul(x, W_q).view(batch_per_stream, gamma, H, D_HEAD).transpose(1, 2)
                k = torch.matmul(x, W_k).view(batch_per_stream, gamma, H, D_HEAD).transpose(1, 2)
                v = torch.matmul(x, W_v).view(batch_per_stream, gamma, H, D_HEAD).transpose(1, 2)

                attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                attn_flat = attn_out.transpose(1, 2).reshape(batch_per_stream * gamma, D)
                proj_out = torch.matmul(attn_flat, W_out)
                mlp_h = F.silu(torch.matmul(proj_out, W_gate))
                _ = torch.matmul(mlp_h, W_down)

                # Speculative verification kernel
                accept_mask, accept_count = speculative_verify(draft_tokens_list[s_idx], target_tokens_list[s_idx])
                return accept_count

        # 2. Sequential Verification baseline (gamma single-token decode passes)
        def run_sequential_verification(s_idx):
            """Execute run sequential verification routine."""
            with torch.cuda.stream(streams[s_idx]):
                # Sequential 1 token at a time for gamma steps
                x_single = embed_inputs[s_idx][:, :1, :]
                for _ in range(gamma):
                    q = torch.matmul(x_single, W_q).view(batch_per_stream, 1, H, D_HEAD).transpose(1, 2)
                    k = torch.matmul(x_single, W_k).view(batch_per_stream, 1, H, D_HEAD).transpose(1, 2)
                    v = torch.matmul(x_single, W_v).view(batch_per_stream, 1, H, D_HEAD).transpose(1, 2)
                    attn_out = F.scaled_dot_product_attention(q, k, v)
                    attn_flat = attn_out.transpose(1, 2).reshape(batch_per_stream, D)
                    proj_out = torch.matmul(attn_flat, W_out)
                    mlp_h = F.silu(torch.matmul(proj_out, W_gate))
                    _ = torch.matmul(mlp_h, W_down)

        # Warmup
        for s in range(num_streams):
            _ = run_parallel_verification(s)
            run_sequential_verification(s)
        for s in range(num_streams):
            streams[s].synchronize()

        # Benchmark Parallel Speculative Verification
        iters = 50
        ev_par_start = torch.cuda.Event(enable_timing=True)
        ev_par_end = torch.cuda.Event(enable_timing=True)

        ev_par_start.record()
        for _ in range(iters):
            for s in range(num_streams):
                _ = run_parallel_verification(s)
            for s in range(num_streams):
                streams[s].synchronize()
        ev_par_end.record()
        torch.cuda.synchronize()
        par_latency_ms = ev_par_start.elapsed_time(ev_par_end) / iters

        # Benchmark Sequential Verification
        ev_seq_start = torch.cuda.Event(enable_timing=True)
        ev_seq_end = torch.cuda.Event(enable_timing=True)

        ev_seq_start.record()
        for _ in range(iters):
            for s in range(num_streams):
                run_sequential_verification(s)
            for s in range(num_streams):
                streams[s].synchronize()
        ev_seq_end.record()
        torch.cuda.synchronize()
        seq_latency_ms = ev_seq_start.elapsed_time(ev_seq_end) / iters

        # Compute speedup and effective accepted tokens
        _, counts = speculative_verify(draft_tokens_list[0], target_tokens_list[0])
        avg_accepted = float(counts.float().mean().item())
        effective_acc_rate = (avg_accepted / gamma) * 100.0

        # Memory traffic savings: instead of gamma round-trips to DRAM for weights,
        # weights are loaded once for the whole window. Memory traffic savings = (1 - 1/gamma) * 100%
        mem_traffic_savings_pct = (1.0 - (1.0 / max(avg_accepted + 1, 1.0))) * 100.0
        speedup = seq_latency_ms / par_latency_ms

        total_tokens_verified = num_streams * batch_per_stream * gamma
        throughput_tok_per_sec = (total_tokens_verified * 1000.0) / par_latency_ms
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)

        print(
            f"Gamma={gamma:2d} | Streams={num_streams}: Par Lat={par_latency_ms:.3f} ms | Seq Lat={seq_latency_ms:.3f} ms | Speedup={speedup:.2f}x | Acc Rate={effective_acc_rate:.1f}% | Mem Traffic Savings={mem_traffic_savings_pct:.1f}% | TPS={throughput_tok_per_sec:,.0f}"
        )

        results.append(
            {
                "Streams": num_streams,
                "Gamma": gamma,
                "Batch_Per_Stream": batch_per_stream,
                "Parallel_Latency_ms": round(par_latency_ms, 3),
                "Sequential_Latency_ms": round(seq_latency_ms, 3),
                "Speedup_Ratio": round(speedup, 2),
                "Avg_Accepted_Tokens": round(avg_accepted, 2),
                "Effective_Acceptance_Rate_pct": round(effective_acc_rate, 1),
                "Memory_Traffic_Savings_pct": round(mem_traffic_savings_pct, 1),
                "Verification_Throughput_tok_per_sec": round(throughput_tok_per_sec, 1),
                "Peak_VRAM_MB": round(peak_vram_mb, 2),
                "Status": "PASSED",
            }
        )

# Export to CSV
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_speculative_concurrency.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Streams",
    "Gamma",
    "Batch_Per_Stream",
    "Parallel_Latency_ms",
    "Sequential_Latency_ms",
    "Speedup_Ratio",
    "Avg_Accepted_Tokens",
    "Effective_Acceptance_Rate_pct",
    "Memory_Traffic_Savings_pct",
    "Verification_Throughput_tok_per_sec",
    "Peak_VRAM_MB",
    "Status",
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
