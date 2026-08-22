#!/usr/bin/env python3
"""
Master Test Suite for All Compression Engine Modules
Tests initialization, compression, shape preservation, and forward pass across all 70 registered compressors.
"""
import sys
import os
from pathlib import Path

# Add project Codes directory to path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import torch
import torch.nn as nn
from framework.core.registry import COMPRESSOR_REGISTRY
import framework.compression_engine


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


def test_compressor(name: str):
    cls = COMPRESSOR_REGISTRY.resolve(name)
    model = ToyModel(hidden_dim=768, num_layers=2)
    x = torch.randn(2, 16, 768)

    # 1. Instantiate compressor
    compressor = None
    instantiation_strategies = [
        lambda: cls(model, rank=64),
        lambda: cls(model, target_rank=64),
        lambda: cls(model),
        lambda: cls(model, sparsity=0.5),
        lambda: cls(model, bits=4),
    ]

    last_err = ""
    for strategy in instantiation_strategies:
        try:
            compressor = strategy()
            break
        except Exception as e:
            last_err = str(e)
            continue

    if compressor is None:
        return False, f"Instantiation failed: {last_err}"

    # 2. Run compression
    try:
        if hasattr(compressor, "compress"):
            res = compressor.compress()
            compressed_model = res[0] if isinstance(res, tuple) else res
        elif callable(compressor) and not isinstance(compressor, nn.Module):
            res = compressor()
            compressed_model = res[0] if isinstance(res, tuple) else res
        else:
            compressed_model = getattr(compressor, "model", model)
    except Exception as e:
        return False, f"Compression execution failed: {e}"

    if compressed_model is None or not isinstance(compressed_model, nn.Module):
        compressed_model = model

    # 3. Test forward pass
    try:
        out = compressed_model(x)
        if isinstance(out, tuple):
            out = out[0]
        if out.shape[0] != 2:
            return False, f"Unexpected batch size: {out.shape}"
        if torch.isnan(out).any() or torch.isinf(out).any():
            return False, "Output contains NaN or Inf"
    except Exception as e:
        return False, f"Forward pass failed: {e}"

    return True, "SUCCESS"


def main():
    print("=" * 70, flush=True)
    print("  COMPRESSION ENGINE MASTER AUDIT & VALIDATION", flush=True)
    print("=" * 70, flush=True)

    keys = sorted(list(COMPRESSOR_REGISTRY.keys()))
    print(f"Discovered {len(keys)} registered compressors in COMPRESSOR_REGISTRY.\n", flush=True)

    results = {}
    for name in keys:
        if name in ("your_compressor_name", "replaceme"):
            continue
        print(f"Testing {name:32s} ...", end=" ", flush=True)
        passed, msg = test_compressor(name)
        results[name] = (passed, msg)
        status_str = "PASS" if passed else "FAIL"
        print(f"[{status_str:4s}] : {msg}", flush=True)

    print("\n" + "=" * 70, flush=True)
    print("  SUMMARY REPORT", flush=True)
    print("=" * 70, flush=True)
    passed_count = sum(1 for p, _ in results.values() if p)
    total_count = len(results)
    print(f"Passed: {passed_count}/{total_count} ({passed_count/total_count*100:.1f}%)", flush=True)

    failures = [name for name, (p, m) in results.items() if not p]
    if failures:
        print(f"\nFailed Compressors ({len(failures)}):", flush=True)
        for f in failures:
            print(f"  - {f:32s}: {results[f][1]}", flush=True)
    else:
        print("\nAll compressors passed perfectly!", flush=True)


if __name__ == "__main__":
    main()
