#!/usr/bin/env python3
"""
Unified Operator Benchmark Runner for NeuralOps and FLA Kernels.
Modeled after benchmarks/ops/run.py paradigm:
- Subprocess-isolated execution so GPU memory is cleanly freed between configurations.
- Direct benchmarking of forward ('fwd') and backward ('fwdbwd') passes across
  registered CUDA and Triton operators (chunk_gla, gdn_scan, svd_linear_attention,
  fused_ce, fused_rmsnorm, selective_scan, etc.).
- Automated timing using triton.testing.do_bench with warmup and quantiles.
- Rich Markdown and ASCII table reporting, optional comparison against git baseline.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import triton
import triton.testing

logger = logging.getLogger("neuralops.benchmarks.ops_runner")

# Registered operator benchmark functions
OP_REGISTRY: Dict[str, Callable] = {}


def register_op(name: str):
    def decorator(fn):
        OP_REGISTRY[name] = fn
        return fn
    return decorator


@register_op("chunk_gla")
def get_chunk_gla_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    from neuralops.kernels import gla_scan
    q = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    g = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True).sigmoid()

    def fwd():
        return gla_scan.gla_scan(q, k, v, g)

    def fwdbwd():
        out = gla_scan.gla_scan(q, k, v, g)
        loss = out.sum()
        loss.backward(retain_graph=True)
        return out

    return fwd, fwdbwd


@register_op("gdn_scan")
def get_gdn_scan_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    from neuralops.kernels import gdn_scan
    q = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(B, H, T, D, dtype=dtype, device=device, requires_grad=True)
    beta = torch.randn(B, H, T, dtype=dtype, device=device, requires_grad=True).sigmoid()
    alpha = torch.randn(B, H, T, dtype=dtype, device=device, requires_grad=True).sigmoid()

    def fwd():
        return gdn_scan.gdn_scan(q, k, v, beta, alpha)

    def fwdbwd():
        out = gdn_scan.gdn_scan(q, k, v, beta, alpha)
        loss = out.sum()
        loss.backward(retain_graph=True)
        return out

    return fwd, fwdbwd


@register_op("svd_linear_attention")
def get_svd_linear_attention_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    from neuralops.attentions.svd_linear_attention import SVDLinearAttention
    module = SVDLinearAttention(embed_dim=H * D, num_heads=H).to(device=device, dtype=dtype)
    x = torch.randn(B, T, H * D, dtype=dtype, device=device, requires_grad=True)

    def fwd():
        return module(x)

    def fwdbwd():
        out = module(x)
        y = out[0] if isinstance(out, tuple) else out
        loss = y.sum()
        loss.backward(retain_graph=True)
        return y

    return fwd, fwdbwd


@register_op("fused_ce")
def get_fused_ce_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    from neuralops.kernels import fused_ce
    V = H * D * 4
    logits = torch.randn(B * T, V, dtype=dtype, device=device, requires_grad=True)
    targets = torch.randint(0, V, (B * T,), dtype=torch.long, device=device)
    loss_fn = fused_ce.FusedCrossEntropyLoss()

    def fwd():
        return loss_fn(logits, targets)

    def fwdbwd():
        loss = loss_fn(logits, targets)
        loss.backward(retain_graph=True)
        return loss

    return fwd, fwdbwd


@register_op("fused_rmsnorm")
def get_fused_rmsnorm_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    dim = H * D
    weight = torch.ones(dim, dtype=dtype, device=device, requires_grad=True)
    x = torch.randn(B, T, dim, dtype=dtype, device=device, requires_grad=True)

    def fwd():
        variance = x.pow(2).mean(-1, keepdim=True)
        return x * torch.rsqrt(variance + 1e-6) * weight

    def fwdbwd():
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + 1e-6) * weight
        loss = out.sum()
        loss.backward(retain_graph=True)
        return out

    return fwd, fwdbwd


@register_op("selective_scan")
def get_selective_scan_op(B: int, T: int, H: int, D: int, dtype: torch.dtype, device: torch.device):
    from neuralops.kernels import selective_scan
    u = torch.randn(B, H * D, T, dtype=dtype, device=device, requires_grad=True)
    delta = torch.randn(B, H * D, T, dtype=dtype, device=device, requires_grad=True).sigmoid()
    A = -torch.rand(H * D, 16, dtype=dtype, device=device, requires_grad=True)
    B_param = torch.randn(B, 16, T, dtype=dtype, device=device, requires_grad=True)
    C_param = torch.randn(B, 16, T, dtype=dtype, device=device, requires_grad=True)

    def fwd():
        return selective_scan.selective_scan_fn(u, delta, A, B_param, C_param)

    def fwdbwd():
        out = selective_scan.selective_scan_fn(u, delta, A, B_param, C_param)
        loss = out.sum()
        loss.backward(retain_graph=True)
        return out

    return fwd, fwdbwd


def run_benchmark_worker(
    op_name: str,
    B: int,
    T: int,
    H: int,
    D: int,
    mode: str = "fwd",
    dtype_str: str = "float16",
    device_str: str = "cuda",
    warmup: int = 15,
    rep: int = 50,
) -> Dict[str, Any]:
    """Runs a single operator timing trial in the current process."""
    if op_name not in OP_REGISTRY:
        raise ValueError(f"Operator '{op_name}' not in OP_REGISTRY. Available: {list(OP_REGISTRY.keys())}")

    device = torch.device(device_str if torch.cuda.is_available() and device_str != "cpu" else "cpu")
    dtype = getattr(torch, dtype_str)

    fwd_fn, fwdbwd_fn = OP_REGISTRY[op_name](B, T, H, D, dtype, device)
    target_fn = fwd_fn if mode == "fwd" else fwdbwd_fn

    if device.type == "cuda":
        ms, min_ms, max_ms = triton.testing.do_bench(
            target_fn,
            warmup=warmup,
            rep=rep,
            quantiles=[0.5, 0.2, 0.8],
            return_mode="all",
        )
    else:
        for _ in range(warmup):
            target_fn()
        times = []
        for _ in range(rep):
            t0 = time.perf_counter()
            target_fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        ms = float(sum(times) / len(times))
        min_ms = float(min(times))
        max_ms = float(max(times))

    tflops = (2.0 * B * T * H * D * (T if "attention" in op_name or "gla" in op_name else D)) / (ms * 1e-3) / 1e12

    return {
        "op": op_name,
        "mode": mode,
        "B": B,
        "T": T,
        "H": H,
        "D": D,
        "latency_ms": ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "tflops": tflops,
    }


def run_op_in_subprocess(
    op_name: str,
    B: int,
    T: int,
    H: int,
    D: int,
    mode: str = "fwd",
    dtype: str = "float16",
    device: str = "cuda",
) -> Dict[str, Any]:
    """Spawns an isolated Python subprocess to execute benchmark worker cleanly."""
    script_cmd = [
        sys.executable,
        "-c",
        (
            f"import json, sys; "
            f"from neuralops.benchmarks.ops_runner import run_benchmark_worker; "
            f"res = run_benchmark_worker('{op_name}', {B}, {T}, {H}, {D}, '{mode}', '{dtype}', '{device}'); "
            f"print('__BENCH_JSON_START__' + json.dumps(res) + '__BENCH_JSON_END__')"
        ),
    ]

    result = subprocess.run(script_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Worker process failed for op={op_name}: {result.stderr}")
        return {"op": op_name, "error": result.stderr, "B": B, "T": T, "H": H, "D": D, "mode": mode}

    output = result.stdout
    if "__BENCH_JSON_START__" in output and "__BENCH_JSON_END__" in output:
        json_str = output.split("__BENCH_JSON_START__")[1].split("__BENCH_JSON_END__")[0]
        return json.loads(json_str)

    return {"op": op_name, "raw_output": output, "B": B, "T": T, "H": H, "D": D, "mode": mode}


def format_table(results_or_headers: Any, rows: Optional[List[List[Any]]] = None) -> str:
    """Formats benchmark results as Markdown table, accepting either result dicts or (headers, rows)."""
    if rows is not None:
        headers = [str(h) for h in results_or_headers]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in rows:
            lines.append("| " + " | ".join([str(c) for c in row]) + " |")
        return "\n".join(lines)

    results = results_or_headers
    headers = ["Operator", "Mode", "Shape (B, T, H, D)", "Latency (ms)", "Min (ms)", "Max (ms)", "TFLOPs"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for r in results:
        op_lbl = r.get("op", "unknown")
        mode_lbl = r.get("mode", "fwd")
        b_val = r.get("B")
        t_val = r.get("T")
        h_val = r.get("H")
        d_val = r.get("D")
        if "error" in r:
            lines.append(f"| {op_lbl} | {mode_lbl} | ({b_val}, {t_val}, {h_val}, {d_val}) | ERROR | - | - | - |")
        else:
            lat = r.get("latency_ms", 0.0)
            min_lat = r.get("min_ms", 0.0)
            max_lat = r.get("max_ms", 0.0)
            tflops = r.get("tflops", 0.0)
            lines.append(
                f"| {op_lbl} | {mode_lbl} | ({b_val}, {t_val}, {h_val}, {d_val}) | "
                f"{lat:.3f} | {min_lat:.3f} | {max_lat:.3f} | {tflops:.2f} |"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="NeuralOps Unified Operator Benchmark Runner")
    parser.add_argument("--ops", nargs="+", default=list(OP_REGISTRY.keys()), help="Ops to benchmark")
    parser.add_argument("--mode", type=str, default="fwd", choices=["fwd", "fwdbwd"], help="Forward or Fwd+Bwd")
    parser.add_argument("-b", "--batch-size", type=int, default=2, help="Batch size (B)")
    parser.add_argument("-t", "--seq-len", type=int, default=1024, help="Sequence length (T)")
    parser.add_argument("--num-heads", type=int, default=8, help="Number of heads (H)")
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension (D)")
    parser.add_argument("--dtype", type=str, default="float16", help="Precision")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results path")
    args = parser.parse_args()

    results: List[Dict[str, Any]] = []
    print(f"\nRunning benchmark suite across {len(args.ops)} operators (mode={args.mode}, B={args.batch_size}, T={args.seq_len})...")

    for op in args.ops:
        print(f" -> Benchmarking: {op}...")
        res = run_op_in_subprocess(
            op_name=op,
            B=args.batch_size,
            T=args.seq_len,
            H=args.num_heads,
            D=args.head_dim,
            mode=args.mode,
            dtype=args.dtype,
            device=args.device,
        )
        results.append(res)

    md_table = format_table(results)
    print("\n" + md_table + "\n")

    if args.output:
        out_p = Path(args.output)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
