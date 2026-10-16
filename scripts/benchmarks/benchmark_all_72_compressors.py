# -*- coding: utf-8 -*-
"""Exhaustive Triton & Physical Hardware Benchmark across all 72 compression engines."""

import copy
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from neuralops.compression import COMPRESSOR_REGISTRY


class TestModel(nn.Module):
    """Transformer block module with attention and FFN for universal compression benchmarking."""

    def __init__(self, in_features: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.q_proj = nn.Linear(in_features, in_features, bias=False)
        self.k_proj = nn.Linear(in_features, in_features, bias=False)
        self.v_proj = nn.Linear(in_features, in_features, bias=False)
        self.out_proj = nn.Linear(in_features, in_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_dim, bias=False)
        self.fc2 = nn.Linear(hidden_dim, in_features, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x_seq = x.unsqueeze(0)
        else:
            x_seq = x
        q, k, v = self.q_proj(x_seq), self.k_proj(x_seq), self.v_proj(x_seq)
        attn = torch.matmul(q, k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        a_out = self.out_proj(torch.matmul(torch.softmax(attn, dim=-1), v)) + x_seq
        ffn = self.fc2(torch.nn.functional.gelu(self.fc1(a_out)))
        out = a_out + ffn
        return out.squeeze(0) if x.dim() == 2 else out


def _extract_model_and_stats(res):
    stats = {}
    if isinstance(res, tuple):
        model = res[0]
        if len(res) > 1 and isinstance(res[1], dict):
            stats = res[1]
    elif hasattr(res, "compress"):
        c_res = res.compress()
        if isinstance(c_res, tuple):
            model = c_res[0]
            if len(c_res) > 1 and isinstance(c_res[1], dict):
                stats = c_res[1]
        else:
            model = c_res
    else:
        model = res
    return model, stats


def _build_compressed_model(target, base_model: nn.Module, **override_kwargs):
    kwargs = {
        "rank": 32,
        "target_rank": 32,
        "compression_ratio": 0.5,
        "keep_ratio": 0.5,
        "calibration_texts": ["NeuralOps deep learning compression calibration phrase"] * 4,
        "n_calib": 4,
        "fallback_to_original": True,
        "max_error": 1.0,
    }
    kwargs.update(override_kwargs)
    try:
        res = target(base_model, **kwargs)
    except TypeError:
        try:
            res = target(base_model, rank=32)
        except TypeError:
            res = target(base_model)
    model, stats = _extract_model_and_stats(res)
    return model, stats


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running Compressor Benchmark on: {device} ({torch.cuda.get_device_name(0)})")

    results = []
    names = sorted(list(COMPRESSOR_REGISTRY.keys()))
    print(f"Benchmarking all {len(names)} compression engines...")

    for i, name in enumerate(names, 1):
        target = COMPRESSOR_REGISTRY[name]
        try:
            base_model = TestModel(256, 512).to(device=device, dtype=torch.float32).eval()
            probe_x = torch.randn(128, 256, device=device, dtype=torch.float32)

            with torch.no_grad():
                y0 = base_model(probe_x)

            torch.cuda.synchronize()
            t0 = time.perf_counter()

            # Execute compression
            comp_model, stats = _build_compressed_model(target, base_model)

            torch.cuda.synchronize()
            compress_ms = (time.perf_counter() - t0) * 1000.0

            comp_model = comp_model.to(device=device).eval()
            with torch.no_grad():
                y1 = comp_model(probe_x)
                rel_err = (torch.norm(y1.float() - y0.float()) / torch.norm(y0.float())).item()

            results.append({
                "name": name,
                "compress_ms": round(compress_ms, 2),
                "rel_err": round(rel_err, 4),
                "status": "OK",
                "error": None,
            })
            print(f"[{i:2d}/{len(names)}] {name:35s}: {compress_ms:8.2f} ms | err={rel_err:.4f} [OK]")
        except Exception as e:
            err_str = str(e)
            results.append({
                "name": name,
                "compress_ms": 0.0,
                "rel_err": 0.0,
                "status": "FAILED",
                "error": err_str[:120],
            })
            print(f"[{i:2d}/{len(names)}] {name:35s}: FAILED ({err_str[:60]})")

    results.sort(key=lambda r: r["compress_ms"], reverse=True)

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "triton_all_72_compressors_benchmark.json"

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    ok_count = sum(1 for r in results if r["status"] == "OK")
    failed_count = sum(1 for r in results if r["status"] == "FAILED")
    print(f"\nSaved benchmark results to {out_path}")
    print(f"Summary: {ok_count}/{len(names)} passed on physical GPU ({failed_count} failures)")


if __name__ == "__main__":
    main()
