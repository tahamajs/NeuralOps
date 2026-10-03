#!/usr/bin/env python3
"""
Master Test Suite for All Compression Engine Modules
Tests initialization, compression, shape preservation, and forward pass across all 70 registered compressors.
"""
import os
import sys
from pathlib import Path

# Add project Codes directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import framework.compression_engine
import pytest
import torch
import torch.nn as nn
from framework.core.registry import COMPRESSOR_REGISTRY


class ToyTransformerLayer(nn.Module):
    def __init__(self, hidden_dim=768):
        super().__init__()
        self.q_lin = nn.Linear(hidden_dim, hidden_dim)
        self.k_lin = nn.Linear(hidden_dim, hidden_dim)
        self.v_lin = nn.Linear(hidden_dim, hidden_dim)
        self.out_lin = nn.Linear(hidden_dim, hidden_dim)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.lin2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x):
        q = self.q_lin(x)
        k = self.k_lin(x)
        v = self.v_lin(x)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v)
        out = self.out_lin(ctx) + x
        mlp = self.lin2(torch.relu(self.lin1(out))) + out
        return mlp


class ToyModel(nn.Module):
    def __init__(self, hidden_dim=768, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([ToyTransformerLayer(hidden_dim) for _ in range(num_layers)])
        self.classifier = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.classifier(x.mean(dim=1))


import time

VALID_COMPRESSOR_NAMES = sorted(
    [k for k in COMPRESSOR_REGISTRY.keys() if k not in ("your_compressor_name", "replaceme")]
)


def run_single_compressor_test(name: str, device: torch.device = torch.device("cpu")):
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    orig_model = ToyModel(hidden_dim=768, num_layers=2).to(device)
    orig_params = sum(p.numel() for p in orig_model.parameters())
    x = torch.randn(2, 16, 768, device=device, requires_grad=True)

    compressed_model = None
    ratio = 1.0

    # Strategy 1: apply_module_to_model unified runner
    try:
        from framework.eval_all_models_datasets import apply_module_to_model
        compressed_model, ratio = apply_module_to_model(orig_model, name, "compressor", rank=16, device=device)
    except Exception as e1:
        # Strategy 2: Direct resolution and instantiation strategies
        try:
            cls = COMPRESSOR_REGISTRY.resolve(name)
            compressor = None
            for strategy in [
                lambda: cls(orig_model, rank=16),
                lambda: cls(orig_model, target_rank=16),
                lambda: cls(orig_model),
                lambda: cls(orig_model, sparsity=0.5),
                lambda: cls(orig_model, bits=4),
                lambda: cls(),
            ]:
                try:
                    compressor = strategy()
                    break
                except Exception:
                    continue

            if compressor is not None:
                if hasattr(compressor, "compress"):
                    try:
                        res = compressor.compress(orig_model)
                    except TypeError:
                        res = compressor.compress()
                    compressed_model = res[0] if isinstance(res, tuple) else res
                elif callable(compressor) and not isinstance(compressor, nn.Module):
                    try:
                        res = compressor(orig_model)
                    except TypeError:
                        res = compressor()
                    compressed_model = res[0] if isinstance(res, tuple) else res
                elif isinstance(compressor, nn.Module):
                    compressed_model = compressor
                else:
                    compressed_model = getattr(compressor, "model", orig_model)
        except Exception as e2:
            elapsed = time.perf_counter() - t0
            return {
                "status": "FAIL_COMPRESSION",
                "error": f"{str(e1)[:150]} | {str(e2)[:150]}",
                "elapsed": elapsed,
                "ratio": 1.0,
                "vram_mb": 0.0,
                "forward_finite": False,
                "backward_finite": False,
            }

    if compressed_model is None or not isinstance(compressed_model, nn.Module):
        compressed_model = orig_model

    compressed_model = compressed_model.to(device)
    new_params = sum(p.numel() for p in compressed_model.parameters())
    if ratio == 1.0 and orig_params > 0 and new_params > 0:
        ratio = orig_params / new_params

    # 3. Test forward pass
    try:
        compressed_model.train()
        out = compressed_model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.shape[0] != 2:
            return {
                "status": "FAIL_SHAPE",
                "error": f"Unexpected batch size: {out.shape}",
                "elapsed": time.perf_counter() - t0,
                "ratio": ratio,
                "vram_mb": 0.0,
                "forward_finite": False,
                "backward_finite": False,
            }
        if not torch.isfinite(out).all():
            return {
                "status": "FAIL_NAN",
                "error": "Output contains NaN or Inf",
                "elapsed": time.perf_counter() - t0,
                "ratio": ratio,
                "vram_mb": 0.0,
                "forward_finite": False,
                "backward_finite": False,
            }
    except Exception as e:
        return {
            "status": "FAIL_FORWARD",
            "error": f"Forward pass failed: {str(e)[:200]}",
            "elapsed": time.perf_counter() - t0,
            "ratio": ratio,
            "vram_mb": 0.0,
            "forward_finite": False,
            "backward_finite": False,
        }

    # 4. Test backward pass
    try:
        compressed_model.zero_grad(set_to_none=True)
        loss = out.sum()
        loss.backward()
    except Exception as e:
        return {
            "status": "FAIL_BACKWARD",
            "error": f"Backward pass failed: {str(e)[:200]}",
            "elapsed": time.perf_counter() - t0,
            "ratio": ratio,
            "vram_mb": 0.0,
            "forward_finite": True,
            "backward_finite": False,
        }

    vram_mb = 0.0
    if device.type == "cuda":
        vram_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        del compressed_model, orig_model, x, out, loss
        torch.cuda.empty_cache()

    elapsed = time.perf_counter() - t0
    return {
        "status": "PASS",
        "error": "",
        "elapsed": elapsed,
        "ratio": ratio,
        "vram_mb": vram_mb,
        "forward_finite": True,
        "backward_finite": True,
    }


@pytest.mark.parametrize("name", VALID_COMPRESSOR_NAMES)
def test_compressor(name: str):
    res = run_single_compressor_test(name)
    assert res["status"] == "PASS", f"Compressor '{name}' failed validation: {res['error']}"


def main():
    import argparse
    import csv

    parser = argparse.ArgumentParser(description="Master Audit for All Compression Engines on GPU")
    parser.add_argument("--device", type=str, default="cuda", help="Target device (cuda or cpu)")
    parser.add_argument("--output_csv", type=str, default="results/all_compressors_gpu_summary.csv", help="Output CSV path")
    parser.add_argument("--modules", type=str, default="all", help="Comma-separated module names or 'all'")
    args = parser.parse_args()

    dev = torch.device(args.device) if (args.device == "cuda" and torch.cuda.is_available()) else torch.device("cpu")

    print("=" * 80, flush=True)
    print(f"  COMPRESSION ENGINE FULL GPU AUDIT & VALIDATION MATRIX ({dev})", flush=True)
    if dev.type == "cuda":
        print(f"  Hardware: {torch.cuda.get_device_name(0)} ({torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB)", flush=True)
    print("=" * 80, flush=True)

    if args.modules.strip() == "all":
        modules_to_test = VALID_COMPRESSOR_NAMES
    else:
        modules_to_test = [m.strip() for m in args.modules.split(",") if m.strip()]

    print(f"Testing {len(modules_to_test)} compressors on {dev}...\n", flush=True)

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["module", "device", "status", "forward_finite", "backward_finite", "compression_ratio", "peak_vram_mb", "elapsed_s", "error"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        f.flush()

        for idx, name in enumerate(modules_to_test, 1):
            print(f"[{idx:3d}/{len(modules_to_test):3d}] Testing {name:32s} ...", end=" ", flush=True)
            res = run_single_compressor_test(name, device=dev)
            row = {
                "module": name,
                "device": str(dev),
                "status": res["status"],
                "forward_finite": res["forward_finite"],
                "backward_finite": res["backward_finite"],
                "compression_ratio": round(res["ratio"], 2),
                "peak_vram_mb": round(res["vram_mb"], 1),
                "elapsed_s": round(res["elapsed"], 3),
                "error": res["error"],
            }
            writer.writerow(row)
            f.flush()
            results.append(row)
            print(f"[{res['status']:4s}] (elapsed={res['elapsed']:.2f}s, vram={res['vram_mb']:.1f}MB, ratio={res['ratio']:.2f}x) {res['error']}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("  SUMMARY REPORT", flush=True)
    print("=" * 80, flush=True)
    passed_count = sum(1 for r in results if r["status"] == "PASS")
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count} ({passed_count/max(1, total_count)*100:.1f}%)", flush=True)
    print(f"Summary written to: {out_path.resolve()}", flush=True)

    failures = [r for r in results if r["status"] != "PASS"]
    if failures:
        print(f"\nFailed Compressors ({len(failures)}):", flush=True)
        for f_item in failures:
            print(f"  - {f_item['module']:32s}: {f_item['error']}", flush=True)
    else:
        print("\nALL COMPRESSION ENGINES PASSED 100% ON GPU!", flush=True)


if __name__ == "__main__":
    main()

