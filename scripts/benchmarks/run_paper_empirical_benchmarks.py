# -*- coding: utf-8 -*-
# Author: Taha Majlesi
"""
Comprehensive Empirical Benchmark Suite for ICLR 2027 Submission.

Evaluates and logs physical GPU measurements on Ada Lovelace SM 8.9 (NVIDIA RTX 4070 Laptop GPU):
1. Multi-batch scaling (B in [1, 2, 4, 8, 16]) and Sequence Length scaling (N in [512, 1024, 2048, 4096, 8192]).
2. Multi-precision numerical parity (FP16, BF16, FP32) with strict error bounds.
3. Peak allocated VRAM tracking via torch.cuda.max_memory_allocated().
4. End-to-end model surgery throughput (Tokens/sec and latency) comparing standard MHA against linear attention.
5. Ingestion of results directly into ResultsLedger (SQLite + JSONL).
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F

from neuralops.core.results_ledger import ResultsLedger, ResultRecord
from neuralops.core.triton_kernels.gla_scan import gla_scan
from neuralops.core.triton_kernels.gdn_scan import gdn_scan
from neuralops.core.triton_kernels.cosformer_triton_kernel import triton_cosformer_linear_attention
import neuralops
from transformers import GPT2Config, GPT2LMHeadModel


def measure_peak_memory_and_latency(fn, warmup: int = 5, iters: int = 20) -> Tuple[float, float]:
    """Measures mean execution time (ms) and peak allocated VRAM (MB)."""
    assert callable(fn), "fn must be callable"
    torch.cuda.synchronize()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    for _ in range(warmup):
        _ = fn()
    torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(iters):
        _ = fn()
    end_event.record()
    torch.cuda.synchronize()

    mean_ms = start_event.elapsed_time(end_event) / iters
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    return mean_ms, peak_vram_mb


def run_scaling_benchmarks(device: str = "cuda") -> List[Dict[str, Any]]:
    """Runs sequence length and batch size scaling sweeps."""
    print("=" * 80)
    print("1. RUNNING CONTEXT LENGTH & MULTI-BATCH SCALING BENCHMARKS")
    print("=" * 80)

    seq_lens = [512, 1024, 2048, 4096, 8192]
    batch_sizes = [1, 2, 4, 8, 16]
    H, D = 12, 64
    scaling_records = []

    # Sequence Length Scaling at B=2
    B_fixed = 2
    for N in seq_lens:
        q = torch.randn(B_fixed, H, N, D, device=device, dtype=torch.float16)
        k = torch.randn(B_fixed, H, N, D, device=device, dtype=torch.float16)
        v = torch.randn(B_fixed, H, N, D, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(B_fixed, H, N, device=device, dtype=torch.float16)).clamp(1e-4, 1.0 - 1e-4)

        # Baseline SDPA
        ms_sdpa, vram_sdpa = measure_peak_memory_and_latency(
            lambda _q=q, _k=k, _v=v: F.scaled_dot_product_attention(_q, _k, _v)
        )

        # Chunk GLA
        ms_gla, vram_gla = measure_peak_memory_and_latency(lambda _q=q, _k=k, _v=v, _g=g: gla_scan(_q, _k, _v, _g))

        # Chunk GDN
        beta = torch.sigmoid(torch.randn(B_fixed, H, N, device=device, dtype=torch.float16))
        ms_gdn, vram_gdn = measure_peak_memory_and_latency(
            lambda _q=q, _k=k, _v=v, _beta=beta, _g=g: gdn_scan(_q, _k, _v, _beta, _g)
        )

        # Cosformer
        ms_cos, vram_cos = measure_peak_memory_and_latency(
            lambda _q=q, _k=k, _v=v: triton_cosformer_linear_attention(_q, _k, _v)
        )

        record = {
            "type": "seq_len_scaling",
            "batch_size": B_fixed,
            "seq_len": N,
            "sdpa_ms": round(ms_sdpa, 3),
            "sdpa_vram_mb": round(vram_sdpa, 1),
            "gla_ms": round(ms_gla, 3),
            "gla_vram_mb": round(vram_gla, 1),
            "gdn_ms": round(ms_gdn, 3),
            "gdn_vram_mb": round(vram_gdn, 1),
            "cosformer_ms": round(ms_cos, 3),
            "cosformer_vram_mb": round(vram_cos, 1),
            "gla_speedup_vs_sdpa": round(ms_sdpa / max(ms_gla, 1e-5), 2),
            "vram_reduction_pct": round((1.0 - vram_gla / max(vram_sdpa, 1e-5)) * 100.0, 1),
        }
        scaling_records.append(record)
        print(
            f"  N={N:4d} | SDPA: {ms_sdpa:6.2f}ms ({vram_sdpa:5.1f}MB) | GLA: {ms_gla:6.2f}ms ({vram_gla:5.1f}MB, {record['gla_speedup_vs_sdpa']}x)"
        )

    # Batch Size Scaling at N=1024
    N_fixed = 1024
    for B in batch_sizes:
        q = torch.randn(B, H, N_fixed, D, device=device, dtype=torch.float16)
        k = torch.randn(B, H, N_fixed, D, device=device, dtype=torch.float16)
        v = torch.randn(B, H, N_fixed, D, device=device, dtype=torch.float16)
        g = torch.sigmoid(torch.randn(B, H, N_fixed, device=device, dtype=torch.float16)).clamp(1e-4, 1.0 - 1e-4)

        ms_sdpa, vram_sdpa = measure_peak_memory_and_latency(
            lambda _q=q, _k=k, _v=v: F.scaled_dot_product_attention(_q, _k, _v)
        )
        ms_gla, vram_gla = measure_peak_memory_and_latency(lambda _q=q, _k=k, _v=v, _g=g: gla_scan(_q, _k, _v, _g))

        record = {
            "type": "batch_scaling",
            "batch_size": B,
            "seq_len": N_fixed,
            "sdpa_ms": round(ms_sdpa, 3),
            "sdpa_vram_mb": round(vram_sdpa, 1),
            "gla_ms": round(ms_gla, 3),
            "gla_vram_mb": round(vram_gla, 1),
            "tok_per_sec": round((B * N_fixed) / (ms_gla / 1000.0), 0),
        }
        scaling_records.append(record)
        print(f"  B={B:2d} | SDPA: {ms_sdpa:6.2f}ms | GLA: {ms_gla:6.2f}ms ({record['tok_per_sec']:7.0f} tok/s)")

    return scaling_records


def run_precision_parity_benchmarks(device: str = "cuda") -> List[Dict[str, Any]]:
    """Evaluates numerical parity across precision formats: FP16, BF16, and FP32."""
    print("\n" + "=" * 80)
    print("2. RUNNING MULTI-PRECISION NUMERICAL PARITY (FP16 vs BF16 vs FP32)")
    print("=" * 80)

    precision_records = []
    B, H, N, D = 2, 4, 512, 64

    for dtype, dtype_name in [(torch.float32, "FP32"), (torch.float16, "FP16"), (torch.bfloat16, "BF16")]:
        q = torch.randn(B, H, N, D, device=device, dtype=dtype)
        k = torch.randn(B, H, N, D, device=device, dtype=dtype)
        v = torch.randn(B, H, N, D, device=device, dtype=dtype)
        g = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype)).clamp(1e-4, 1.0 - 1e-4)

        out_triton = gla_scan(q, k, v, g)

        # PyTorch reference computed in FP32
        q_32, k_32, v_32, g_32 = q.float(), k.float(), v.float(), g.float()
        from neuralops.core.triton_kernels.gla_scan import gla_scan_pytorch

        out_ref = gla_scan_pytorch(q_32, k_32, v_32, g_32).to(dtype=dtype)

        diff = (out_triton.float() - out_ref.float()).abs()
        max_abs_diff = diff.max().item()
        cos_sim = F.cosine_similarity(out_triton.float().flatten(), out_ref.float().flatten(), dim=0).item()
        nan_count = torch.isnan(out_triton).sum().item()

        record = {
            "operator": "gla_scan",
            "precision": dtype_name,
            "max_abs_diff": max_abs_diff,
            "cos_sim": cos_sim,
            "nan_count": nan_count,
            "parity_pass": bool(max_abs_diff < 0.01 and nan_count == 0),
        }
        precision_records.append(record)
        print(
            f"  {dtype_name:4s} | Max Diff: {max_abs_diff:.6e} | Cos Sim: {cos_sim:.6f} | NaNs: {nan_count} | Status: {'PASS' if record['parity_pass'] else 'FAIL'}"
        )

    return precision_records


def run_e2e_model_surgery_benchmarks(device: str = "cuda") -> List[Dict[str, Any]]:
    """Evaluates full transformer model surgery throughput and peak VRAM."""
    print("\n" + "=" * 80)
    print("3. RUNNING END-TO-END MODEL SURGERY BENCHMARK (GPT-2)")
    print("=" * 80)

    cfg = GPT2Config(n_embd=768, n_layer=12, n_head=12, vocab_size=50257, n_positions=8192)
    base_mha = GPT2LMHeadModel(cfg).to(device=device, dtype=torch.float16).eval()
    base_svd = GPT2LMHeadModel(cfg).to(device=device, dtype=torch.float16).eval()
    linear_model = neuralops.replace_attention(base_svd, attention="positive_kernel_linear_attention").eval()

    e2e_records = []
    configurations = [
        (1, 512),
        (2, 512),
        (4, 512),
        (1, 1024),
        (2, 1024),
        (4, 1024),
        (1, 2048),
        (2, 2048),
    ]

    for B, N in configurations:
        x = torch.randint(0, 50257, (B, N), device=device)

        # Baseline MHA
        ms_mha, vram_mha = measure_peak_memory_and_latency(lambda _x=x: base_mha(_x), warmup=3, iters=10)
        tok_mha = (B * N) / (ms_mha / 1000.0)

        # Linear Attention Surgery
        ms_lin, vram_lin = measure_peak_memory_and_latency(lambda _x=x: linear_model(_x), warmup=3, iters=10)
        tok_lin = (B * N) / (ms_lin / 1000.0)

        speedup = ms_mha / ms_lin
        rec = {
            "batch_size": B,
            "seq_len": N,
            "mha_latency_ms": round(ms_mha, 2),
            "mha_vram_mb": round(vram_mha, 1),
            "mha_tok_per_sec": round(tok_mha, 0),
            "linear_latency_ms": round(ms_lin, 2),
            "linear_vram_mb": round(vram_lin, 1),
            "linear_tok_per_sec": round(tok_lin, 0),
            "speedup": round(speedup, 2),
        }
        e2e_records.append(rec)
        print(
            f"  B={B:2d}, N={N:4d} | MHA: {ms_mha:6.2f}ms ({tok_mha:7.0f} tok/s) | Linear: {ms_lin:6.2f}ms ({tok_lin:7.0f} tok/s) | Speedup: {speedup:4.2f}x"
        )

    return e2e_records


def ingest_records_into_ledger(scaling_records, precision_records, e2e_records, out_dir: Path):
    """Ingests all benchmark metrics into ResultsLedger."""
    print("\n" + "=" * 80)
    print("4. INGESTING BENCHMARK RESULTS INTO MASTER RESULTS LEDGER")
    print("=" * 80)

    ledger = ResultsLedger(results_dir=out_dir)

    # 1. Ingest Scaling Records
    for rec in scaling_records:
        r = ResultRecord(
            record_id="",
            module_name="gla_scan" if rec["type"] == "seq_len_scaling" else "gla_scan_batch",
            category="triton_kernel",
            family="linear_attention",
            version_tag="v1.3.7",
            status="active",
            experimental_setup={
                "batch_size": rec["batch_size"],
                "seq_len": rec["seq_len"],
                "hardware": torch.cuda.get_device_name(0),
            },
            metrics=rec,
            notes=f"Physical GPU context scaling measurement ({rec['type']})",
        )
        ledger.add_or_replace(r)

    # 2. Ingest E2E Surgery Records
    for rec in e2e_records:
        r = ResultRecord(
            record_id="",
            module_name="positive_kernel_linear_attention_gpt2",
            category="end_to_end",
            family="model_surgery",
            version_tag="v1.3.7",
            status="active",
            experimental_setup={
                "backbone": "gpt2",
                "batch_size": rec["batch_size"],
                "seq_len": rec["seq_len"],
                "hardware": torch.cuda.get_device_name(0),
            },
            metrics=rec,
            notes="End-to-end GPT-2 model surgery benchmark vs Dense MHA",
        )
        ledger.add_or_replace(r)

    print("Successfully ingested records into master ResultsLedger.")


def main():
    """Execute main routine."""
    if not torch.cuda.is_available():
        print("CUDA is unavailable. Cannot proceed.")
        sys.exit(1)

    device = "cuda:0"
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    scaling_records = run_scaling_benchmarks(device)
    precision_records = run_precision_parity_benchmarks(device)
    e2e_records = run_e2e_model_surgery_benchmarks(device)

    all_data = {
        "scaling_records": scaling_records,
        "precision_records": precision_records,
        "e2e_records": e2e_records,
        "device": torch.cuda.get_device_name(0),
    }

    out_file = out_dir / "paper_empirical_benchmarks_master.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    print(f"\nSaved empirical benchmark master results to {out_file}")

    ingest_records_into_ledger(scaling_records, precision_records, e2e_records, out_dir)


if __name__ == "__main__":
    main()
