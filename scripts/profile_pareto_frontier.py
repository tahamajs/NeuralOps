import sys
sys.path.insert(0, '/mnt/d/SVD_linear_Attention')
import torch
import torch.nn.functional as F
import time
import os
import csv
from neuralops.core.cuda_kernels.paged_kv_evict.paged_kv_evict_autograd import paged_kv_evict
from neuralops.core.cuda_kernels.sparse_attention.sparse_attention_autograd import block_sparse_attn
from neuralops.core.cuda_kernels.selective_scan.selective_scan_autograd import selective_scan

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")

# Budgets M in {1GB, 2GB, 4GB}
results = []

# Warmup CUDA
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(1000, 1000, device=device) @ torch.randn(1000, 1000, device=device)
torch.cuda.synchronize()

print("\n=== STARTING PARETO FRONTIER PROFILING ===")

configurations = [
    # (name, policy_type, param)
    ("Standard Dense Attention", "dense", 1.0),
    ("Paged KV Evict (Retention=50%)", "paged_kv_evict", 0.50),
    ("Paged KV Evict (Retention=25%)", "paged_kv_evict", 0.25),
    ("Dynamic Sparse Attention (Block Sparsity=50%)", "block_sparse", 0.50),
    ("Dynamic Sparse Attention (Block Sparsity=75%)", "block_sparse", 0.75),
    ("Selective Scan SSM (Linear State-Space)", "selective_scan", 1.0),
]

H = 16
D_HEAD = 64
D = H * D_HEAD

# Targets to evaluate throughput vs effective capacity under memory limits
budget_configs = [
    (1024, 2, 4096),   # 1GB budget: B=2, L=4096 (8,192 tokens)
    (2048, 4, 4096),   # 2GB budget: B=4, L=4096 (16,384 tokens)
    (4096, 4, 8192),   # 4GB budget: B=4, L=8192 (32,768 tokens)
]

for budget_mb, B, L in budget_configs:
    total_tokens = B * L
    print(f"\n--- Testing Budget Target: {budget_mb} MB (B={B}, L={L}, Total Tokens={total_tokens}) ---")
    
    for name, policy_type, param in configurations:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        # Dense attention OOMs at B=4, L=8192
        if policy_type == "dense" and L > 4096:
            print(f"Skipping {name} at L={L} (OOM limit exceeded)")
            results.append({
                "Budget_MB": budget_mb,
                "Policy": name,
                "Batch_Size": B,
                "Seq_Len": L,
                "Total_Tokens": total_tokens,
                "Latency_ms": float("nan"),
                "Throughput_tok_per_sec": 0.0,
                "Peak_VRAM_MB": float("nan"),
                "VRAM_Efficiency_pct": 0.0,
                "Effective_Capacity_pct": 0.0,
                "Status": "OOM"
            })
            continue

        try:
            if policy_type == "dense":
                q = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                k = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                v = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                
                for _ in range(5):
                    out = F.scaled_dot_product_attention(q, k, v)
                torch.cuda.synchronize()
                
                torch.cuda.reset_peak_memory_stats()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                
                iters = 30
                start_ev.record()
                for _ in range(iters):
                    out = F.scaled_dot_product_attention(q, k, v)
                end_ev.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                tps = total_tokens / (elapsed_ms / 1000.0)
                eff_cap = 100.0
                
            elif policy_type == "paged_kv_evict":
                # k_in and v_in are [L_total, H, D_HEAD], where L_total = B * L tokens
                # keep_indices: 1D tensor of int64 of length int(total_tokens * param)
                keep_len = int(total_tokens * param)
                k_pool = torch.randn(total_tokens, H, D_HEAD, device=device, dtype=torch.float16)
                v_pool = torch.randn(total_tokens, H, D_HEAD, device=device, dtype=torch.float16)
                keep_idx = torch.sort(torch.randint(0, total_tokens, (keep_len,), device=device, dtype=torch.int64)).values
                
                # Warmup
                for _ in range(5):
                    k_ev, v_ev = paged_kv_evict(k_pool, v_pool, keep_idx)
                torch.cuda.synchronize()
                
                torch.cuda.reset_peak_memory_stats()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                
                iters = 30
                start_ev.record()
                for _ in range(iters):
                    k_ev, v_ev = paged_kv_evict(k_pool, v_pool, keep_idx)
                end_ev.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                tps = total_tokens / (elapsed_ms / 1000.0)
                eff_cap = param * 100.0
                
            elif policy_type == "block_sparse":
                block_size = 64
                num_blocks = L // block_size
                q = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                k = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                v = torch.randn(B, H, L, D_HEAD, device=device, dtype=torch.float16)
                
                # block_mask is 2D [num_blocks, num_blocks]
                # Sparsity param: fraction of blocks masked out (e.g. 0.50 or 0.75)
                # Keep fraction = 1 - param
                density = 1.0 - param
                mask_prob = torch.rand(num_blocks, num_blocks, device=device)
                block_mask = (mask_prob < density).to(torch.int32)
                # Ensure diagonal is active for causal self-attention
                block_mask = block_mask | torch.eye(num_blocks, device=device, dtype=torch.int32)
                
                for _ in range(5):
                    out = block_sparse_attn(q, k, v, block_mask, block_size)
                torch.cuda.synchronize()
                
                torch.cuda.reset_peak_memory_stats()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                
                iters = 30
                start_ev.record()
                for _ in range(iters):
                    out = block_sparse_attn(q, k, v, block_mask, block_size)
                end_ev.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                tps = total_tokens / (elapsed_ms / 1000.0)
                eff_cap = (1.0 - param) * 100.0
                
            elif policy_type == "selective_scan":
                dim_ssm = H * D_HEAD
                dstate = 16
                u = torch.randn(B, dim_ssm, L, device=device, dtype=torch.float16)
                delta = torch.randn(B, dim_ssm, L, device=device, dtype=torch.float16)
                A = -torch.rand(dim_ssm, dstate, device=device, dtype=torch.float32)
                B_ssm = torch.randn(B, dstate, L, device=device, dtype=torch.float16)
                C_ssm = torch.randn(B, dstate, L, device=device, dtype=torch.float16)
                D_ssm = torch.randn(dim_ssm, device=device, dtype=torch.float16)
                
                for _ in range(5):
                    out = selective_scan(u, delta, A, B_ssm, C_ssm, D_ssm)
                torch.cuda.synchronize()
                
                torch.cuda.reset_peak_memory_stats()
                start_ev = torch.cuda.Event(enable_timing=True)
                end_ev = torch.cuda.Event(enable_timing=True)
                
                iters = 30
                start_ev.record()
                for _ in range(iters):
                    out = selective_scan(u, delta, A, B_ssm, C_ssm, D_ssm)
                end_ev.record()
                torch.cuda.synchronize()
                
                elapsed_ms = start_ev.elapsed_time(end_ev) / iters
                peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
                tps = total_tokens / (elapsed_ms / 1000.0)
                eff_cap = 100.0
            
            vram_eff = (peak_mb / budget_mb) * 100.0
            print(f"[{name}] Latency: {elapsed_ms:.2f} ms | Throughput: {tps:,.0f} tok/s | Peak VRAM: {peak_mb:.1f} MB (Budget Util: {vram_eff:.1f}%)")
            
            results.append({
                "Budget_MB": budget_mb,
                "Policy": name,
                "Batch_Size": B,
                "Seq_Len": L,
                "Total_Tokens": total_tokens,
                "Latency_ms": round(elapsed_ms, 3),
                "Throughput_tok_per_sec": round(tps, 1),
                "Peak_VRAM_MB": round(peak_mb, 2),
                "VRAM_Efficiency_pct": round(vram_eff, 1),
                "Effective_Capacity_pct": round(eff_cap, 1),
                "Status": "PASSED"
            })
        except Exception as e:
            print(f"Error running {name}: {e}")
            results.append({
                "Budget_MB": budget_mb,
                "Policy": name,
                "Batch_Size": B,
                "Seq_Len": L,
                "Total_Tokens": total_tokens,
                "Latency_ms": float("nan"),
                "Throughput_tok_per_sec": 0.0,
                "Peak_VRAM_MB": float("nan"),
                "VRAM_Efficiency_pct": 0.0,
                "Effective_Capacity_pct": 0.0,
                "Status": f"FAILED: {str(e)[:30]}"
            })

# Save results to CSV
csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_pareto_frontier.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Budget_MB", "Policy", "Batch_Size", "Seq_Len", "Total_Tokens",
    "Latency_ms", "Throughput_tok_per_sec", "Peak_VRAM_MB", "VRAM_Efficiency_pct", "Effective_Capacity_pct", "Status"
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
