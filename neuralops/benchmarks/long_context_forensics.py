#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Measure long-context attention scaling against a PyTorch SDPA baseline.

This is an evidence-generating benchmark, not a correctness certificate: the
registry module and SDPA baseline are timed independently and their outputs are
not claimed to be mathematically identical for mechanisms with different
attention definitions.  OOM is recorded explicitly instead of being treated
as an artificial speedup.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from neuralops.core.registry import ATTENTION_REGISTRY, create_attention


TARGETS = (
    "cosformer",
    "gla_attention",
    "gated_linear_attention",
    "mamba_block",
    "selective_ssm",
    "rwkv6",
    "sliding_window",
    "local_attention",
)
SEQUENCE_LENGTHS = (512, 1024, 2048, 4096, 8192)


def _resolve_targets() -> list[str]:
    """Return requested mechanisms that are actually present in this registry."""
    keys = {str(name).lower() for name in ATTENTION_REGISTRY.keys()}
    selected = [name for name in TARGETS if name in keys]
    if len(selected) < 8:
        patterns = ("linear", "ssm", "mamba", "gla", "cosformer", "rwkv", "local", "sparse")
        for name in sorted(keys):
            if name not in selected and any(pattern in name for pattern in patterns):
                selected.append(name)
            if len(selected) == 8:
                break
    return selected[:8]


def _unwrap(value: Any) -> torch.Tensor:
    """Extract a tensor from common NeuralOps attention return conventions."""
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)) and value and isinstance(value[0], torch.Tensor):
        return value[0]
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state
    raise TypeError(f"Attention returned unsupported value: {type(value)!r}")


def _timed(fn: Any, warmups: int = 3, repeats: int = 10) -> tuple[float, float]:
    """Return median CUDA latency and peak allocated VRAM in MiB."""
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    samples: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    samples.sort()
    return samples[len(samples) // 2], float(torch.cuda.max_memory_allocated() / (1024**2))


def _sdpa_reference(x: torch.Tensor, heads: int = 8) -> torch.Tensor:
    """Run standard scaled dot-product attention over a fixed Q/K/V projection."""
    batch, seq, hidden = x.shape
    head_dim = hidden // heads
    q = x.view(batch, seq, heads, head_dim).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, q, q, is_causal=False)
    return out.transpose(1, 2).reshape(batch, seq, hidden)


def run_module(name: str, sequence: int, device: str = "cuda") -> dict[str, Any]:
    """Run one module/sequence pair and preserve OOM or construction failures."""
    torch.manual_seed(42)
    dtype = torch.float16
    batch, hidden, heads = 4, 512, 8
    record: dict[str, Any] = {
        "module": name,
        "category": "attention",
        "sequence_length": sequence,
        "batch": batch,
        "hidden": hidden,
        "heads": heads,
        "dtype": str(dtype),
    }
    try:
        x = torch.randn(batch, sequence, hidden, device=device, dtype=dtype)
        module = create_attention(name, embed_dim=hidden, num_heads=heads).to(device=device, dtype=dtype).eval()
        fast_ms, peak_mb = _timed(lambda: _unwrap(module(x)))
        record.update({"cuda_latency_ms": fast_ms, "peak_vram_mb": peak_mb})
        try:
            ref_ms, ref_peak = _timed(lambda: _sdpa_reference(x, heads=heads))
            record.update(
                {
                    "pytorch_latency_ms": ref_ms,
                    "pytorch_peak_vram_mb": ref_peak,
                    "speedup_ratio": ref_ms / fast_ms if fast_ms > 0 else 0.0,
                    "status": "PASS_FAST" if ref_ms / fast_ms >= 1.05 else "PASS_SLOW",
                }
            )
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            record.update(
                {
                    "pytorch_latency_ms": None,
                    "pytorch_peak_vram_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
                    "speedup_ratio": None,
                    "status": "OOM_PYTORCH",
                    "error_reason": str(exc),
                }
            )
            torch.cuda.empty_cache()
    except (torch.cuda.OutOfMemoryError, RuntimeError, TypeError, ValueError) as exc:
        record.update({"status": "ERROR", "error_reason": str(exc)})
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/long_context_forensics.json"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA is required for long-context forensics")
    targets = _resolve_targets()
    rows: list[dict[str, Any]] = []
    for name in targets:
        for sequence in SEQUENCE_LENGTHS:
            started = time.time()
            row = run_module(name, sequence, args.device)
            row["wall_time_s"] = round(time.time() - started, 3)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"targets": targets, "rows": rows}, indent=2) + "\n")


if __name__ == "__main__":
    main()
