#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unified Operator Benchmark Runner for NeuralOps and FLA Kernels

Modeled after benchmarks/ops/run.py paradigm.
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


def run_op_as_eval_result(
    op_name: str,
    B: int = 2,
    T: int = 1024,
    H: int = 8,
    D: int = 64,
    mode: str = "fwd",
    dtype: str = "float16",
    device: str = "cuda",
) -> Any:
    """Run an operator benchmark and convert output into a standardized BenchmarkEvalResult."""
    from neuralops.benchmarks.eval_bridge import BenchmarkEvalResult

    raw = run_op_in_subprocess(op_name, B, T, H, D, mode, dtype, device)
    status = "FAIL" if "error" in raw else "PASS"
    err = raw.get("error")
    lat = raw.get("latency_ms", 0.0)
    tflops = raw.get("tflops", 0.0)

    total_tokens = B * T
    throughput = (total_tokens / (lat / 1000.0)) if lat > 0 else 0.0

    return BenchmarkEvalResult(
        task_type=f"op_{mode}",
        status=status,
        error_message=err,
        model_name=f"operator_{op_name}",
        attention_type=op_name,
        device=device,
        dtype=dtype,
        batch_size=B,
        seq_len=T,
        total_tokens=total_tokens,
        latency_ms_mean=lat,
        latency_ms_min=raw.get("min_ms", 0.0),
        latency_ms_max=raw.get("max_ms", 0.0),
        latency_ms_per_token=lat / max(1, T),
        throughput_tokens_per_sec=throughput,
        achieved_tflops=tflops,
        extra_metrics=raw,
    )


def run_registered_ops(
    *, category: str = "all", modules: str | list[str] | None = None,
    output: str | Path = "results/benchmark_ops.json", verify_only: bool = False,
    verbose: bool = False, dynamic: bool = False, timeout_s: float = 180.0,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Run registered operations in killable subprocesses with checkpointing.

    A benchmark operation may enter a native extension or a CUDA launch that
    does not return control to the parent Python process.  Running each
    operation in a child process makes the timeout enforceable: the parent can
    terminate the child and persist a ``TIMEOUT`` row without losing earlier
    results.  Existing rows in ``output`` are treated as completed work, which
    makes rerunning the same command a safe resume operation.
    """
    import os
    import tempfile
    from neuralops.benchmarks.ops_registry import list_ops

    names = [m.strip() for m in modules.split(",") if m.strip()] if isinstance(modules, str) else modules
    # Registry discovery must be a set-like operation.  A stale registry or
    # merged checkpoint must never schedule the same operation twice.
    selected = list(dict.fromkeys(names or list_ops(category)))
    target = Path(output)
    target.parent.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    if target.exists():
        try:
            previous = json.loads(target.read_text(encoding="utf-8"))
            if isinstance(previous, list):
                # Keep the first row for each module so an old duplicated
                # checkpoint cannot inflate the release total on resume.
                seen: set[str] = set()
                for row in previous:
                    if not isinstance(row, dict):
                        continue
                    module = str(row.get("module", ""))
                    if not module or module in seen:
                        continue
                    seen.add(module)
                    results.append(row)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Ignoring unreadable checkpoint %s: %s", target, exc)
    completed = {str(row.get("module")) for row in results}

    for name in selected:
        if name in completed:
            if verbose:
                print(f"[RESUME] {name}", flush=True)
            continue
        if dynamic:
            while True:
                try:
                    import psutil
                    free_ram = psutil.virtual_memory().available
                except ImportError:
                    free_ram = 2**63
                free_vram = 2**63
                try:
                    query = subprocess.run(
                        ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
                        capture_output=True, text=True, check=True,
                    )
                    free_vram = float(query.stdout.strip().splitlines()[0]) * 1024**2
                except (OSError, subprocess.SubprocessError, ValueError, IndexError):
                    pass
                if free_ram >= 1.5 * 1024**3 and free_vram >= 1000 * 1024**2:
                    break
                if verbose:
                    print(f"PAUSING 4s: RAM={free_ram / 1024**3:.2f}GB VRAM={free_vram / 1024**2:.0f}MB", flush=True)
                time.sleep(4)
        worker_fd, worker_name = tempfile.mkstemp(
            prefix="neuralops_bench_", suffix=".json", dir=target.parent
        )
        os.close(worker_fd)
        worker_path = Path(worker_name)
        worker_code = (
            "import sys; "
            "from neuralops.benchmarks.ops_runner import _run_registered_op_worker; "
            "_run_registered_op_worker(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])"
        )
        proc = subprocess.Popen(
                [sys.executable, "-c", worker_code, name, str(verify_only), str(worker_path), device or ""],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        started_at = time.monotonic()
        try:
            _, stderr = proc.communicate(timeout=max(1.0, timeout_s))
            if proc.returncode != 0:
                row = {
                    "module": name,
                    "category": "unknown",
                    "cuda_latency_ms": 0.0,
                    "pytorch_latency_ms": 0.0,
                    "speedup": 0.0,
                    "peak_vram_mb": 0.0,
                    "max_diff": float("inf"),
                    "status": "FAIL",
                    "error_reason": stderr[-2000:] if stderr else f"worker exit {proc.returncode}",
                }
            else:
                row = json.loads(worker_path.read_text(encoding="utf-8"))
        except subprocess.TimeoutExpired:
            proc.kill()
            _, stderr = proc.communicate(timeout=10)
            row = {
                "module": name,
                "category": "unknown",
                "cuda_latency_ms": 0.0,
                "pytorch_latency_ms": 0.0,
                "speedup": 0.0,
                "peak_vram_mb": 0.0,
                "max_diff": float("inf"),
                "status": "TIMEOUT",
                "error_reason": f"worker exceeded {timeout_s:.0f}s and was killed",
            }
            logger.warning("TIMEOUT %s after %.1fs", name, time.monotonic() - started_at)
        except (OSError, json.JSONDecodeError) as exc:
            row = {
                "module": name,
                "category": "unknown",
                "cuda_latency_ms": 0.0,
                "pytorch_latency_ms": 0.0,
                "speedup": 0.0,
                "peak_vram_mb": 0.0,
                "max_diff": float("inf"),
                "status": "FAIL",
                "error_reason": str(exc),
            }
        finally:
            worker_path.unlink(missing_ok=True)
        results.append(row)
        completed.add(name)
        if verbose:
            print(f"[{row['status']}] {name}", flush=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def _run_registered_op_worker(op_name: str, verify_only: str, output_path: str, device: str = "") -> None:
    """Execute one registered operation and write a JSON row for the parent."""
    from neuralops.benchmarks.ops_registry import get_op
    from neuralops.benchmarks.verify_ops import _make_inputs, _tensor_output, verify_op

    config = get_op(op_name)
    row: dict[str, Any] = {
        "module": op_name,
        "category": config.category,
        "cuda_latency_ms": 0.0,
        "pytorch_latency_ms": 0.0,
        "speedup": 0.0,
        "peak_vram_mb": 0.0,
        "max_diff": float("inf"),
        "status": "FAIL_CORRECTNESS",
        "error_reason": config.unavailable_reason,
    }

    def measure(fn: Callable[[], Any]) -> float:
        for _ in range(5):
            fn()
        if torch.cuda.is_available():
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(25):
                fn()
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end) / 25)
        started = time.perf_counter()
        for _ in range(25):
            fn()
        return (time.perf_counter() - started) * 1000.0 / 25

    requested_device = device or None
    if verify_op(op_name, device=requested_device):
        if verify_only.lower() == "true":
            row.update(status="VERIFIED", max_diff=0.0, error_reason=None)
        else:
            args, kwargs = _make_inputs(config, requested_device)
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            fast_ms = measure(lambda: config.func(*args, **kwargs))
            ref_ms = measure(lambda: config.ref_func(*args, **kwargs))
            fast = _tensor_output(config.func(*args, **kwargs))
            ref = _tensor_output(config.ref_func(*args, **kwargs))
            diff = float((fast.float() - ref.float()).abs().max().item())
            ratio = ref_ms / fast_ms if fast_ms > 0 else 0.0
            row.update(
                cuda_latency_ms=fast_ms,
                pytorch_latency_ms=ref_ms,
                speedup=ratio,
                peak_vram_mb=(
                    float(torch.cuda.max_memory_allocated() / (1024**2))
                    if torch.cuda.is_available() else 0.0
                ),
                max_diff=diff,
                status=(
                    "PASS_HIGH_DIFF" if config.category == "attention" and 1e-2 < diff < 0.25
                    else "PASS_FAST" if ratio >= 1.05 else "PASS_SLOW"
                ),
                error_reason=None,
            )
    Path(output_path).write_text(json.dumps(row), encoding="utf-8")


if __name__ == "__main__":
    main()
