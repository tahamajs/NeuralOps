#!/usr/bin/env python3
"""
Isolated benchmark for the 5 slowest kernels on physical GPU.
Measures latency with L2 cache flushing and evaluates autotuning configs.
"""
import sys
import time
import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is required for this benchmark.")

device = torch.device("cuda:0")
dtype = torch.float16

# L2 Cache Flusher (256MB)
num_floats = (256 * 1024 * 1024) // 4
flusher = torch.empty(num_floats, dtype=torch.float32, device=device)

def flush():
    flusher.zero_()
    torch.cuda.synchronize(device)

def measure(fn, *args, repeat=20, warmup=5, **kwargs):
    for _ in range(warmup):
        _ = fn(*args, **kwargs)
    torch.cuda.synchronize(device)
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeat):
        flush()
        start_event.record()
        _ = fn(*args, **kwargs)
        end_event.record()
        torch.cuda.synchronize(device)
        times.append(start_event.elapsed_time(end_event))
    return sum(times) / len(times)

print("=" * 70)
print(f"Profiling 5 Targeted Kernels on: {torch.cuda.get_device_name(0)}")
print("=" * 70)

# 1. fused_favor_plus
from neuralops.ops.triton.elementwise import fused_favor_plus_feature_map_pytorch, fused_favor_plus_feature_map_triton
x_fav = torch.randn(4, 8, 256, 64, device=device, dtype=torch.float32)
om_fav = torch.randn(128, 64, device=device, dtype=torch.float32)
t_pt_1 = measure(fused_favor_plus_feature_map_pytorch, x_fav, om_fav, num_features=128, kernel_type="favor_plus")
t_tri_1 = measure(fused_favor_plus_feature_map_triton, x_fav, om_fav, num_features=128, kernel_type="favor_plus")
sp_1 = t_pt_1 / max(1e-6, t_tri_1)
print(f"1. fused_favor_plus:         PT={t_pt_1:.3f} ms | Triton={t_tri_1:.3f} ms | Speedup={sp_1:.2f}x")

# 2. fused_jamba_hybrid_gate
from neuralops.ops.triton.selective_scan import fused_jamba_hybrid_ssm_attn_gate_pytorch, fused_jamba_hybrid_ssm_attn_gate_triton
attn_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
ssm_out_j = torch.randn(8, 512, 1024, device=device, dtype=torch.float32)
alpha_j = torch.randn(16, device=device, dtype=torch.float32)
t_pt_2 = measure(fused_jamba_hybrid_ssm_attn_gate_pytorch, attn_out_j, ssm_out_j, alpha_j)
t_tri_2 = measure(fused_jamba_hybrid_ssm_attn_gate_triton, attn_out_j, ssm_out_j, alpha_j)
sp_2 = t_pt_2 / max(1e-6, t_tri_2)
print(f"2. fused_jamba_hybrid_gate:  PT={t_pt_2:.3f} ms | Triton={t_tri_2:.3f} ms | Speedup={sp_2:.2f}x")

# 3. fused_moba_block_router
from neuralops.ops.triton.block_sparse import fused_moba_block_router_combine_pytorch, fused_moba_block_router_combine_triton
Qb_m = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
Kb_m = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
Vb_m = torch.randn(4, 4, 16, 16, 32, device=device, dtype=torch.float32)
t_pt_3 = measure(fused_moba_block_router_combine_pytorch, Qb_m, Kb_m, Vb_m, 0.25, top_k=2, is_causal=True)
t_tri_3 = measure(fused_moba_block_router_combine_triton, Qb_m, Kb_m, Vb_m, 0.25, top_k=2, is_causal=True)
sp_3 = t_pt_3 / max(1e-6, t_tri_3)
print(f"3. fused_moba_block_router:  PT={t_pt_3:.3f} ms | Triton={t_tri_3:.3f} ms | Speedup={sp_3:.2f}x")

# 4. fused_norm_linear
from neuralops.ops.triton.fused_norm_linear import fused_rmsnorm_gate, fused_rmsnorm_gate_triton
dim = 2048
x_fnl = torch.randn(2, 512, dim, device=device, dtype=dtype)
g_fnl = torch.randn(2, 512, dim, device=device, dtype=dtype)
w_fnl = torch.randn(dim, device=device, dtype=dtype)
t_pt_4 = measure(fused_rmsnorm_gate, x_fnl, g_fnl, w_fnl)
t_tri_4 = measure(fused_rmsnorm_gate_triton, x_fnl, g_fnl, w_fnl)
sp_4 = t_pt_4 / max(1e-6, t_tri_4)
print(f"4. fused_norm_linear:        PT={t_pt_4:.3f} ms | Triton={t_tri_4:.3f} ms | Speedup={sp_4:.2f}x")

# 5. paged_attention
from neuralops.ops.triton.paged_attention import paged_attention_pytorch, paged_attention_kernel
num_blocks, block_size, num_heads, head_dim = 128, 16, 4, 64
k_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
v_cache = torch.randn(num_blocks, block_size, num_heads, head_dim, device=device, dtype=dtype)
block_tables = torch.arange(num_blocks, device=device, dtype=torch.int32).view(2, 64)
context_lens = torch.full((2,), 64 * block_size, device=device, dtype=torch.int32)
q_pa = torch.randn(2, num_heads, head_dim, device=device, dtype=dtype)
t_pt_5 = measure(paged_attention_pytorch, q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
t_tri_5 = measure(paged_attention_kernel, q_pa, k_cache, v_cache, block_tables, context_lens, block_size=block_size)
sp_5 = t_pt_5 / max(1e-6, t_tri_5)
print(f"5. paged_attention:          PT={t_pt_5:.3f} ms | Triton={t_tri_5:.3f} ms | Speedup={sp_5:.2f}x")

print("=" * 70)
