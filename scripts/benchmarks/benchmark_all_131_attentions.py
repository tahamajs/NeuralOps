# -*- coding: utf-8 -*-
"""Exhaustive Triton Execution & Timing Benchmark across all 131 registered attention mechanisms."""

import json
import os
import sys
import time
from pathlib import Path

import torch
from neuralops.core.registry import ATTENTION_REGISTRY, create_attention


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on hardware device: {device} ({torch.cuda.get_device_name(0)})")

    B, T, D = 2, 512, 256
    results = []

    names = sorted(list(ATTENTION_REGISTRY.keys()))
    print(f"Benchmarking all {len(names)} attention mechanisms at (B={B}, T={T}, D={D}, FP16)...")

    for i, name in enumerate(names, 1):
        x = torch.randn(B, T, D, device=device, dtype=torch.float16, requires_grad=True)
        try:
            mod = create_attention(name, embed_dim=D, num_heads=4).to(device=device, dtype=torch.float16)
            mod.train()

            # Warmup (forward + backward)
            for _ in range(3):
                out = mod(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = out.sum()
                loss.backward()
                x.grad = None
                for p in mod.parameters():
                    p.grad = None

            # Measured timing (20 iterations)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(20):
                out = mod(x)
                if isinstance(out, tuple):
                    out = out[0]
                loss = out.sum()
                loss.backward()
                x.grad = None
                for p in mod.parameters():
                    p.grad = None
            torch.cuda.synchronize()
            triton_ms = (time.perf_counter() - t0) / 20.0 * 1000.0

            results.append({
                "name": name,
                "latency_ms": round(triton_ms, 3),
                "status": "OK",
                "error": None,
            })
            print(f"[{i:3d}/{len(names)}] {name:38s}: {triton_ms:8.2f} ms [OK]")
        except Exception as e:
            err_str = str(e)
            results.append({
                "name": name,
                "latency_ms": 0.0,
                "status": "FAILED",
                "error": err_str[:120],
            })
            print(f"[{i:3d}/{len(names)}] {name:38s}: FAILED ({err_str[:60]})")

    results.sort(key=lambda r: r["latency_ms"], reverse=True)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triton_all_131_attentions_benchmark.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved benchmark results to {out_path}")
    ok_count = sum(1 for r in results if r["status"] == "OK")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    print(f"Summary: {ok_count}/{len(names)} passed forward+backward on GPU ({failed_count} failures)")


if __name__ == "__main__":
    main()
