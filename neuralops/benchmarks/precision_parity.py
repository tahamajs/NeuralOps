#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Canonical mixed-precision numerical parity benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F

from neuralops.benchmarks.config import BenchmarkConfig
from neuralops.benchmarks.results import BenchmarkResult, save_all_formats


def _attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if hasattr(F, "scaled_dot_product_attention"):
        return F.scaled_dot_product_attention(q, k, v)
    scale = q.shape[-1] ** -0.5
    return torch.softmax(q @ k.transpose(-2, -1) * scale, dim=-1) @ v


def run_precision_parity(config: BenchmarkConfig) -> list[BenchmarkResult]:
    """Compare configured low-precision outputs against an FP32 reference."""
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested for precision parity, but CUDA is unavailable.")
    device = torch.device("cuda" if config.device == "cuda" else config.device)
    results: list[BenchmarkResult] = []
    for seq_len in config.seq_lens:
        shape = (config.batch_sizes[0], 4, seq_len, 64)
        q = torch.randn(*shape, device=device, dtype=torch.float32)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        reference = _attention(q, k, v).float()
        for name in ("float16", "bfloat16"):
            dtype = getattr(torch, name)
            if device.type == "cpu" and dtype == torch.bfloat16:
                # CPU BF16 support varies by PyTorch build; skip only when the operation itself is unavailable.
                try:
                    candidate = _attention(q.to(dtype), k.to(dtype), v.to(dtype)).float()
                except RuntimeError as exc:
                    raise RuntimeError(f"BF16 precision benchmark is unavailable on this CPU: {exc}") from exc
            else:
                candidate = _attention(q.to(dtype), k.to(dtype), v.to(dtype)).float()
            diff = (candidate - reference).abs()
            max_diff = float(diff.max().item())
            relative_error = float((diff / reference.abs().clamp_min(1e-6)).max().item())
            tolerance = 1e-2 if name == "float16" else 2e-2
            status = "success" if max_diff <= tolerance and torch.isfinite(candidate).all() else "FAIL_NUMERICS"
            result = BenchmarkResult(
                module={"name": "scaled_dot_product_attention", "type": "precision", "family": "attention"},
                model={"name": "synthetic_attention_probe"},
                dataset={"name": "synthetic", "task": "precision_parity"},
                config={"device": str(device), "dtype": name, "seq_len": seq_len, "tolerance": tolerance},
                metrics={"max_diff": max_diff, "relative_error": relative_error},
                status=status,
                error=None if status == "success" else f"max_diff={max_diff} exceeds tolerance={tolerance}",
            )
            save_all_formats(result, Path(config.output_dir))
            results.append(result)
    if any(item.status != "success" for item in results):
        raise RuntimeError("Precision parity gate failed; one or more results are FAIL_NUMERICS.")
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    run_precision_parity(BenchmarkConfig.from_file(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
