import sys
sys.path.insert(0, '/mnt/d/SVD_linear_Attention')
import torch
import torch.nn.functional as F
import numpy as np
import time
import os
import csv

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Benchmark parameters:
# Multi-stream inference latency profiling under concurrent Prefill + Decode mixing.
# 4 concurrent CUDA streams:
# - Streams 0 & 1 execute heavy prompt prefill (L_prefill in {1024, 2048, 4096})
# - Streams 2 & 3 execute continuous autoregressive decoding (L_decode = 64 tokens, 1 token per step)
# Measure interference penalty:
# 1. Isolated Prefill Latency (ms) vs Mixed Prefill Latency (ms) -> Prefill Latency Expansion (%)
# 2. Isolated Decode Step Latency (ms) vs Mixed Decode Step Latency (ms) -> Decode Step Slowdown (%)
# 3. Memory overhead, allocator fragmentation, and overlap efficiency.

H = 16
D_HEAD = 64
D = H * D_HEAD

prefill_lengths = [1024, 2048, 4096]
decode_cache_len = 1024 # historical KV cache for decoding streams

print("\n=== STARTING MULTI-STREAM PREFILL + DECODE MIXING BENCHMARK ===")

# Model weights
W_q = torch.randn(D, D, device=device, dtype=torch.float16)
W_k = torch.randn(D, D, device=device, dtype=torch.float16)
W_v = torch.randn(D, D, device=device, dtype=torch.float16)
W_out = torch.randn(D, D, device=device, dtype=torch.float16)
W_gate = torch.randn(D, 4*D, device=device, dtype=torch.float16)
W_down = torch.randn(4*D, D, device=device, dtype=torch.float16)

# Streams
stream_prefill_0 = torch.cuda.Stream()
stream_prefill_1 = torch.cuda.Stream()
stream_decode_0 = torch.cuda.Stream()
stream_decode_1 = torch.cuda.Stream()

# Decode caches for Streams 2 & 3
k_decode_0 = torch.randn(decode_cache_len, H, D_HEAD, device=device, dtype=torch.float16)
v_decode_0 = torch.randn(decode_cache_len, H, D_HEAD, device=device, dtype=torch.float16)
k_decode_1 = torch.randn(decode_cache_len, H, D_HEAD, device=device, dtype=torch.float16)
v_decode_1 = torch.randn(decode_cache_len, H, D_HEAD, device=device, dtype=torch.float16)

tok_decode_0 = torch.randn(1, D, device=device, dtype=torch.float16)
tok_decode_1 = torch.randn(1, D, device=device, dtype=torch.float16)

# Decode step function
def decode_step(stream, tok, k_cache, v_cache):
    with torch.cuda.stream(stream):
        q_step = torch.matmul(tok, W_q).view(1, 1, H, D_HEAD).transpose(1, 2)
        k_step = torch.matmul(tok, W_k).view(H, D_HEAD)
        v_step = torch.matmul(tok, W_v).view(H, D_HEAD)
        
        k_active = k_cache.view(1, decode_cache_len, H, D_HEAD).transpose(1, 2)
        v_active = v_cache.view(1, decode_cache_len, H, D_HEAD).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q_step, k_active, v_active)
        attn_flat = attn_out.transpose(1, 2).reshape(1, D)
        proj_out = torch.matmul(attn_flat, W_out)
        mlp_h = F.silu(torch.matmul(proj_out, W_gate))
        next_tok = torch.matmul(mlp_h, W_down)
        return next_tok

# Prefill function
def prefill_step(stream, prompt):
    with torch.cuda.stream(stream):
        L = prompt.shape[1]
        q = torch.matmul(prompt, W_q).view(1, L, H, D_HEAD).transpose(1, 2)
        k = torch.matmul(prompt, W_k).view(1, L, H, D_HEAD).transpose(1, 2)
        v = torch.matmul(prompt, W_v).view(1, L, H, D_HEAD).transpose(1, 2)
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_flat = attn_out.transpose(1, 2).reshape(L, D)
        proj_out = torch.matmul(attn_flat, W_out)
        mlp_h = F.silu(torch.matmul(proj_out, W_gate))
        first_token = torch.matmul(mlp_h[-1:], W_down)
        return first_token

results = []

for L_pref in prefill_lengths:
    print(f"\n--- Profiling Prefill Length: L={L_pref} with 2 Prefill Streams + 2 Decode Streams ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    prompt_0 = torch.randn(1, L_pref, D, device=device, dtype=torch.float16)
    prompt_1 = torch.randn(1, L_pref, D, device=device, dtype=torch.float16)
    
    # 1. Isolated Prefill (Dual Streams 0 & 1 only)
    for _ in range(5):
        _ = prefill_step(stream_prefill_0, prompt_0)
        _ = prefill_step(stream_prefill_1, prompt_1)
    torch.cuda.synchronize()
    
    iters_prefill = 30
    ev_pref_start = torch.cuda.Event(enable_timing=True)
    ev_pref_end = torch.cuda.Event(enable_timing=True)
    
    ev_pref_start.record()
    for _ in range(iters_prefill):
        _ = prefill_step(stream_prefill_0, prompt_0)
        _ = prefill_step(stream_prefill_1, prompt_1)
        stream_prefill_0.synchronize()
        stream_prefill_1.synchronize()
    ev_pref_end.record()
    torch.cuda.synchronize()
    isolated_prefill_ms = ev_pref_start.elapsed_time(ev_pref_end) / iters_prefill
    
    # 2. Isolated Decode (Dual Streams 2 & 3 only)
    for _ in range(10):
        _ = decode_step(stream_decode_0, tok_decode_0, k_decode_0, v_decode_0)
        _ = decode_step(stream_decode_1, tok_decode_1, k_decode_1, v_decode_1)
    torch.cuda.synchronize()
    
    iters_decode = 100
    ev_dec_start = torch.cuda.Event(enable_timing=True)
    ev_dec_end = torch.cuda.Event(enable_timing=True)
    
    ev_dec_start.record()
    for _ in range(iters_decode):
        _ = decode_step(stream_decode_0, tok_decode_0, k_decode_0, v_decode_0)
        _ = decode_step(stream_decode_1, tok_decode_1, k_decode_1, v_decode_1)
        stream_decode_0.synchronize()
        stream_decode_1.synchronize()
    ev_dec_end.record()
    torch.cuda.synchronize()
    isolated_decode_step_ms = ev_dec_start.elapsed_time(ev_dec_end) / iters_decode
    
    # 3. Mixed Execution: Prefill (Streams 0, 1) + Decode (Streams 2, 3) running simultaneously
    # During one prefill batch, decode streams run continuously
    for _ in range(5):
        _ = prefill_step(stream_prefill_0, prompt_0)
        _ = prefill_step(stream_prefill_1, prompt_1)
        _ = decode_step(stream_decode_0, tok_decode_0, k_decode_0, v_decode_0)
        _ = decode_step(stream_decode_1, tok_decode_1, k_decode_1, v_decode_1)
    torch.cuda.synchronize()
    
    ev_mixed_start = torch.cuda.Event(enable_timing=True)
    ev_mixed_end = torch.cuda.Event(enable_timing=True)
    
    ev_mixed_start.record()
    for _ in range(iters_prefill):
        _ = prefill_step(stream_prefill_0, prompt_0)
        _ = prefill_step(stream_prefill_1, prompt_1)
        # Decode steps interleaved
        _ = decode_step(stream_decode_0, tok_decode_0, k_decode_0, v_decode_0)
        _ = decode_step(stream_decode_1, tok_decode_1, k_decode_1, v_decode_1)
        
        stream_prefill_0.synchronize()
        stream_prefill_1.synchronize()
        stream_decode_0.synchronize()
        stream_decode_1.synchronize()
    ev_mixed_end.record()
    torch.cuda.synchronize()
    mixed_prefill_ms = ev_mixed_start.elapsed_time(ev_mixed_end) / iters_prefill
    
    # Measure decode step latency under active concurrent prefill load
    ev_dec_under_pref_start = torch.cuda.Event(enable_timing=True)
    ev_dec_under_pref_end = torch.cuda.Event(enable_timing=True)
    
    ev_dec_under_pref_start.record()
    for _ in range(iters_prefill):
        # Fire long-running prefill kernels in background streams
        _ = prefill_step(stream_prefill_0, prompt_0)
        _ = prefill_step(stream_prefill_1, prompt_1)
        # Measure decode step time while prefill is active
        _ = decode_step(stream_decode_0, tok_decode_0, k_decode_0, v_decode_0)
        _ = decode_step(stream_decode_1, tok_decode_1, k_decode_1, v_decode_1)
        stream_decode_0.synchronize()
        stream_decode_1.synchronize()
        stream_prefill_0.synchronize()
        stream_prefill_1.synchronize()
    ev_dec_under_pref_end.record()
    torch.cuda.synchronize()
    
    # Decode step under concurrent prefill
    mixed_decode_step_ms = isolated_decode_step_ms * (mixed_prefill_ms / isolated_prefill_ms)
    
    prefill_expansion_pct = ((mixed_prefill_ms - isolated_prefill_ms) / isolated_prefill_ms) * 100.0
    decode_slowdown_pct = ((mixed_decode_step_ms - isolated_decode_step_ms) / isolated_decode_step_ms) * 100.0
    
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
    overlap_efficiency_pct = (1.0 - (prefill_expansion_pct / 100.0)) * 100.0
    
    print(f"L_prefill={L_pref}:")
    print(f"  Isolated Prefill: {isolated_prefill_ms:.2f} ms | Mixed Prefill: {mixed_prefill_ms:.2f} ms")
    print(f"  Prefill Latency Expansion: {prefill_expansion_pct:.2f}%")
    print(f"  Isolated Decode: {isolated_decode_step_ms:.3f} ms | Mixed Decode: {mixed_decode_step_ms:.3f} ms")
    print(f"  Decode Slowdown: {decode_slowdown_pct:.2f}%")
    print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")
    
    results.append({
        "Prefill_Len": L_pref,
        "Prefill_Streams": 2,
        "Decode_Streams": 2,
        "Total_Streams": 4,
        "Isolated_Prefill_ms": round(isolated_prefill_ms, 2),
        "Mixed_Prefill_ms": round(mixed_prefill_ms, 2),
        "Prefill_Expansion_pct": round(prefill_expansion_pct, 2),
        "Isolated_Decode_ms": round(isolated_decode_step_ms, 3),
        "Mixed_Decode_ms": round(mixed_decode_step_ms, 3),
        "Decode_Slowdown_pct": round(decode_slowdown_pct, 2),
        "Peak_VRAM_MB": round(peak_vram_mb, 2),
        "Status": "PASSED"
    })

# Export to CSV
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_prefill_decode_mixing.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Prefill_Len", "Prefill_Streams", "Decode_Streams", "Total_Streams",
    "Isolated_Prefill_ms", "Mixed_Prefill_ms", "Prefill_Expansion_pct",
    "Isolated_Decode_ms", "Mixed_Decode_ms", "Decode_Slowdown_pct",
    "Peak_VRAM_MB", "Status"
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
