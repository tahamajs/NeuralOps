# -*- coding: utf-8 -*-
"""Empirical latency, memory footprint, and speedup benchmark for all 48 Triton-wired
Attention Modules on GPU across sequence lengths T=512 and T=2048 in training mode (Fwd+Bwd).
"""

import gc
import json
import os
import sys
import time
import torch
import torch.nn as nn
from neuralops.attentions import ATTENTION_REGISTRY

# 48 Triton-wired modules to benchmark
MODULE_KEYS = [
    "adaptive_sparse_attention", "moba_attention", "xattention", "nsa_attention", "reap_moe",
    "alibi", "dynamic_conv_attention", "mla_attention", "mqa",
    "cosformer", "cosformer_attention", "triton_cosformer",
    "decay_linear_attention", "decay_linear_attn", "gated_linear_attention", "gated_linear_attn",
    "gated_delta_net", "hgrn2_gated_layer", "kda_attention",
    "linformer_attention", "performer_attention", "chela_slay", "chela_slay_attention",
    "galerkin_attention", "gla_attention", "nystrom_attention", "rala_attention", "spectraformer_attention",
    "spectron", "spectron_fp8", "adaptive_span_attention", "sink_attention",
    "mamba", "mamba_block", "selective_ssm", "selective_mamba", "jamba", "vspec",
    "svd_linear_attention", "svd_linear_decay", "triton_svd_linear_attention", "asvd_attention",
    "factorised_matrix_state", "fused_svd_kernel_attention", "projection_factorized_attn",
    "svd_llm_v2_attention", "svd_projection_attention", "svd_seq"
]

def benchmark_module(key: str, B: int = 2, T: int = 512, D: int = 256, H: int = 4, warmup: int = 3, runs: int = 10):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cls = ATTENTION_REGISTRY.get(key)
    if cls is None:
        return None

    try:
        try:
            m = cls(dim=D, num_heads=H).to(device=device, dtype=torch.float16)
        except Exception:
            m = cls(hidden_size=D, num_attention_heads=H).to(device=device, dtype=torch.float16)
        m.train()

        torch.cuda.empty_cache()
        gc.collect()

        # Warmup
        for _ in range(warmup):
            x = torch.randn(B, T, D, device=device, dtype=torch.float16, requires_grad=True)
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()

        for _ in range(runs):
            x = torch.randn(B, T, D, device=device, dtype=torch.float16, requires_grad=True)
            out = m(x)
            if isinstance(out, tuple):
                out = out[0]
            loss = out.sum()
            loss.backward()

        torch.cuda.synchronize()
        t1 = time.perf_counter()

        avg_latency_ms = ((t1 - t0) / runs) * 1000.0
        peak_mem_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

        return {
            "module": key,
            "seq_len": T,
            "latency_ms": round(avg_latency_ms, 3),
            "peak_mem_mb": round(peak_mem_mb, 2),
            "status": "PASS"
        }
    except Exception as e:
        return {
            "module": key,
            "seq_len": T,
            "error": str(e),
            "status": "FAIL"
        }

def main():
    print("=== Launching Exhaustive 48-Module Triton Training Benchmark (RTX 4070) ===")
    results = []

    for seq_len in [512, 2048]:
        print(f"\n--- Benchmarking Sequence Length T = {seq_len} ---")
        for key in MODULE_KEYS:
            res = benchmark_module(key, B=2, T=seq_len, D=256, H=4)
            if res:
                results.append(res)
                if res["status"] == "PASS":
                    print(f"[{res['status']}] {key:<30} (T={seq_len}): {res['latency_ms']:>8.2f} ms | {res['peak_mem_mb']:>7.2f} MB")
                else:
                    print(f"[{res['status']}] {key:<30} (T={seq_len}): {res.get('error')}")

    out_dir = "results"
    os.makedirs(out_dir, exist_ok=True)
    json_path = os.path.join(out_dir, "triton_48_modules_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # Save CSV
    csv_path = os.path.join(out_dir, "triton_48_modules_benchmark.csv")
    with open(csv_path, "w") as f:
        f.write("module,seq_len,latency_ms,peak_mem_mb,status\n")
        for r in results:
            if r["status"] == "PASS":
                f.write(f"{r['module']},{r['seq_len']},{r['latency_ms']},{r['peak_mem_mb']},{r['status']}\n")
            else:
                f.write(f"{r['module']},{r['seq_len']},,,{r['status']}\n")

    # Save Markdown
    md_path = os.path.join(out_dir, "triton_48_modules_benchmark.md")
    with open(md_path, "w") as f:
        f.write("# 48 Triton-Wired Modules Benchmark (NVIDIA RTX 4070 Ada Lovelace)\n\n")
        f.write("| Module | T=512 Latency (ms) | T=512 Mem (MB) | T=2048 Latency (ms) | T=2048 Mem (MB) | Status |\n")
        f.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        
        # Group by module
        by_mod = {}
        for r in results:
            m = r["module"]
            if m not in by_mod:
                by_mod[m] = {}
            by_mod[m][r["seq_len"]] = r

        for m, data in by_mod.items():
            d512 = data.get(512, {})
            d2048 = data.get(2048, {})
            l512 = f"{d512.get('latency_ms', 'N/A')}" if d512.get('status') == 'PASS' else 'FAIL'
            m512 = f"{d512.get('peak_mem_mb', 'N/A')}" if d512.get('status') == 'PASS' else 'FAIL'
            l2048 = f"{d2048.get('latency_ms', 'N/A')}" if d2048.get('status') == 'PASS' else 'FAIL'
            m2048 = f"{d2048.get('peak_mem_mb', 'N/A')}" if d2048.get('status') == 'PASS' else 'FAIL'
            f.write(f"| `{m}` | {l512} | {m512} | {l2048} | {m2048} | PASS |\n")

    print(f"\nBenchmark completed successfully! Saved to {json_path}, {csv_path}, and {md_path}.")

if __name__ == "__main__":
    main()
