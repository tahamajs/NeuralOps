# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Profile eviction overhead module.
"""

import sys

sys.path.insert(0, "/mnt/d/SVD_linear_Attention")
import torch
import torch.nn.functional as F
import os
import csv
from neuralops.core.cuda_kernels.paged_kv_evict.paged_kv_evict_autograd import paged_kv_evict

device = torch.device("cuda:0")
print(f"Device: {torch.cuda.get_device_name(0)}")


H = 16
D_HEAD = 64
D = H * D_HEAD
NUM_LAYERS = 32
EVICTION_INTERVAL = 64

batch_sizes = [1, 2, 4, 8]
seq_lengths = [2048, 4096, 8192]
retention_ratio = 0.50

print("\n=== STARTING AMORTIZED & MULTI-LAYER DYNAMIC KV CACHE EVICTION OVERHEAD PROFILING ===")

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()
_ = torch.randn(2000, 2000, device=device) @ torch.randn(2000, 2000, device=device)
torch.cuda.synchronize()

results = []

for B in batch_sizes:
    for L in seq_lengths:
        total_tokens = B * L
        print(f"\n--- Profiling B={B}, L={L} (Total Tokens: {total_tokens:,}) ---")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        try:
            q_step = torch.randn(B, H, 1, D_HEAD, device=device, dtype=torch.float16)
            k_cache = torch.randn(total_tokens, H, D_HEAD, device=device, dtype=torch.float16)
            v_cache = torch.randn(total_tokens, H, D_HEAD, device=device, dtype=torch.float16)

            W_proj = torch.randn(D, D, device=device, dtype=torch.float16)
            W_gate = torch.randn(D, 4 * D, device=device, dtype=torch.float16)
            W_down = torch.randn(4 * D, D, device=device, dtype=torch.float16)

            num_keep = int(total_tokens * retention_ratio)
            keep_idx = torch.sort(torch.randint(0, total_tokens, (num_keep,), device=device, dtype=torch.int64)).values

            k_reshaped = k_cache.view(B, L, H, D_HEAD).transpose(1, 2)
            v_reshaped = v_cache.view(B, L, H, D_HEAD).transpose(1, 2)

            def run_single_layer(k, v):
                """Execute run single layer routine."""
                attn_out = F.scaled_dot_product_attention(q_step, k, v)
                attn_flat = attn_out.squeeze(2).reshape(B, D)
                proj_out = torch.matmul(attn_flat, W_proj)
                mlp_h = F.silu(torch.matmul(proj_out, W_gate))
                mlp_out = torch.matmul(mlp_h, W_down)
                return mlp_out

            for _ in range(10):
                _ = run_single_layer(k_reshaped, v_reshaped)
            torch.cuda.synchronize()

            iters = 50
            start_base = torch.cuda.Event(enable_timing=True)
            end_base = torch.cuda.Event(enable_timing=True)
            start_base.record()
            for _ in range(iters):
                _ = run_single_layer(k_reshaped, v_reshaped)
            end_base.record()
            torch.cuda.synchronize()
            single_layer_ms = start_base.elapsed_time(end_base) / iters

            model_step_ms = single_layer_ms * NUM_LAYERS

            for _ in range(10):
                _ = paged_kv_evict(k_cache, v_cache, keep_idx)
            torch.cuda.synchronize()

            start_evict = torch.cuda.Event(enable_timing=True)
            end_evict = torch.cuda.Event(enable_timing=True)
            start_evict.record()
            for _ in range(iters):
                k_ev, v_ev = paged_kv_evict(k_cache, v_cache, keep_idx)
            end_evict.record()
            torch.cuda.synchronize()
            eviction_ms = start_evict.elapsed_time(end_evict) / iters

            amortized_eviction_per_step_ms = (eviction_ms * NUM_LAYERS) / EVICTION_INTERVAL
            overhead_pct = (amortized_eviction_per_step_ms / model_step_ms) * 100.0

            peak_vram_mb = torch.cuda.max_memory_allocated() / (1024**2)
            passes_threshold = overhead_pct < 3.5
            status = "PASSED (<3.5%)" if passes_threshold else "EXCEEDED"

            print(f"B={B}, L={L}:")
            print(f"  Single Layer Step: {single_layer_ms:.4f} ms | 32-Layer Forward Step: {model_step_ms:.2f} ms")
            print(f"  Eviction Kernel Time: {eviction_ms:.4f} ms")
            print(f"  Amortized Eviction Cost (W={EVICTION_INTERVAL}): {amortized_eviction_per_step_ms:.4f} ms/step")
            print(f"  Eviction Overhead Ratio: {overhead_pct:.2f}% (Threshold < 3.5%: {status})")
            print(f"  Peak VRAM: {peak_vram_mb:.1f} MB")

            results.append(
                {
                    "Batch_Size": B,
                    "Seq_Len": L,
                    "Total_Tokens": total_tokens,
                    "Single_Layer_ms": round(single_layer_ms, 4),
                    "Model_32L_Step_ms": round(model_step_ms, 2),
                    "Eviction_Kernel_ms": round(eviction_ms, 4),
                    "Amortized_Eviction_ms": round(amortized_eviction_per_step_ms, 4),
                    "Eviction_Overhead_pct": round(overhead_pct, 2),
                    "Peak_VRAM_MB": round(peak_vram_mb, 2),
                    "Status": status,
                }
            )

        except Exception as e:
            print(f"Error B={B}, L={L}: {e}")
            results.append(
                {
                    "Batch_Size": B,
                    "Seq_Len": L,
                    "Total_Tokens": total_tokens,
                    "Single_Layer_ms": float("nan"),
                    "Model_32L_Step_ms": float("nan"),
                    "Eviction_Kernel_ms": float("nan"),
                    "Amortized_Eviction_ms": float("nan"),
                    "Eviction_Overhead_pct": float("nan"),
                    "Peak_VRAM_MB": float("nan"),
                    "Status": f"OOM/FAILED: {str(e)[:30]}",
                }
            )

csv_path = "/mnt/d/SVD_linear_Attention/docs/benchmarks/paper_benchmarks_eviction_overhead.csv"
os.makedirs(os.path.dirname(csv_path), exist_ok=True)
fieldnames = [
    "Batch_Size",
    "Seq_Len",
    "Total_Tokens",
    "Single_Layer_ms",
    "Model_32L_Step_ms",
    "Eviction_Kernel_ms",
    "Amortized_Eviction_ms",
    "Eviction_Overhead_pct",
    "Peak_VRAM_MB",
    "Status",
]

with open(csv_path, mode="w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"\n✅ Results exported successfully to {csv_path}")
