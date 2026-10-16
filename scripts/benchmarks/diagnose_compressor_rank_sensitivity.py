#!/usr/bin/env python3
"""
Forensic diagnostic script to verify rank-sensitivity across all 26 compressors.
Classifies each compressor into:
  1. VALID: rank changes the compressed weights and error (legitimate low-rank SVD / rank-sensitive)
  2. NO_OP_FALLBACK: rejected compression and fell back to original weights (error=0.0)
  3. NO_OP_IGNORES_RANK: compressed weights at rank=8 and rank=64 are identical (ignores rank parameter)
  4. FAILED: crashes or constructor fails
"""
import os
import sys
import copy
import logging
import torch
import torch.nn as nn
from pathlib import Path

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("RankSensitivityDiagnostic")

from scripts.benchmarks.batch_validate_all_compressors import (
    COMPRESSOR_REGISTRY_KEYS,
    load_synthetic_block,
    load_roberta,
    load_gpt2,
    get_probe_data,
    get_compressor,
    evaluate,
)

def extract_weight_signature(model: nn.Module) -> torch.Tensor:
    """Concatenate sample weights into a single flat vector for exact difference check."""
    params = []
    for p in model.parameters():
        if p.ndim >= 2 and p.numel() > 100:
            params.append(p.data.detach().reshape(-1))
    if not params:
        for p in model.parameters():
            params.append(p.data.detach().reshape(-1))
    return torch.cat(params).float()

def run_diagnostic():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running Rank Sensitivity Diagnostic on %s", device)

    torch.manual_seed(42)
    anchor = load_synthetic_block(device)
    probe, heldout, _, _ = get_probe_data("synth_block", device)
    W_orig = extract_weight_signature(anchor)
    y_anchor = anchor(heldout)

    results = {
        "VALID": [],
        "NO_OP_IGNORES_RANK": [],
        "NO_OP_FALLBACK": [],
        "FAILED": [],
    }

    print("\n" + "=" * 105)
    print(f"{'Method':<28s} {'Status':<22s} {'Diff(r8,r64)':<14s} {'Diff(r8,orig)':<14s} {'Err(r8) -> Err(r64)':<22s}")
    print("=" * 105)

    for method in COMPRESSOR_REGISTRY_KEYS:
        # Test rank 8
        m8 = copy.deepcopy(anchor)
        comp8 = get_compressor(method, m8, 8, probe)
        if comp8 is None:
            results["FAILED"].append((method, "Constructor returned None"))
            print(f"{method:<28s} {'FAILED (None)':<22s} {'—':<14s} {'—':<14s} {'—':<22s}")
            continue

        try:
            if hasattr(comp8, "compress"):
                res = comp8.compress()
                if isinstance(res, tuple) and res and isinstance(res[0], nn.Module):
                    m8 = res[0]
                elif isinstance(res, nn.Module):
                    m8 = res
            elif isinstance(comp8, nn.Module):
                m8 = comp8
            elif getattr(comp8, "model", None) is not None:
                m8 = comp8.model
            
            w8 = extract_weight_signature(m8)
            out8 = evaluate(m8, heldout, y_anchor, "synth_block", device)
            err8 = out8["rel_err"]
        except Exception as e:
            results["FAILED"].append((method, str(e)[:40]))
            print(f"{method:<28s} {'FAILED (Exec r8)':<22s} {'—':<14s} {'—':<14s} {'—':<22s}")
            continue

        # Test rank 64
        m64 = copy.deepcopy(anchor)
        comp64 = get_compressor(method, m64, 64, probe)
        if comp64 is None:
            results["FAILED"].append((method, "Rank 64 constructor failed"))
            print(f"{method:<28s} {'FAILED (r64 None)':<22s} {'—':<14s} {'—':<14s} {'—':<22s}")
            continue

        try:
            if hasattr(comp64, "compress"):
                res = comp64.compress()
                if isinstance(res, tuple) and res and isinstance(res[0], nn.Module):
                    m64 = res[0]
                elif isinstance(res, nn.Module):
                    m64 = res
            elif isinstance(comp64, nn.Module):
                m64 = comp64
            elif getattr(comp64, "model", None) is not None:
                m64 = comp64.model

            w64 = extract_weight_signature(m64)
            out64 = evaluate(m64, heldout, y_anchor, "synth_block", device)
            err64 = out64["rel_err"]
        except Exception as e:
            results["FAILED"].append((method, str(e)[:40]))
            print(f"{method:<28s} {'FAILED (Exec r64)':<22s} {'—':<14s} {'—':<14s} {'—':<22s}")
            continue

        diff_8_64 = (w8 - w64).abs().max().item()
        diff_8_orig = (w8 - W_orig).abs().max().item()
        err_str = f"{err8:.4f} -> {err64:.4f}"

        if diff_8_orig < 1e-6:
            status = "NO_OP_FALLBACK"
            results["NO_OP_FALLBACK"].append((method, err_str))
        elif diff_8_64 < 1e-6:
            status = "NO_OP_IGNORES_RANK"
            results["NO_OP_IGNORES_RANK"].append((method, err_str))
        else:
            status = "VALID"
            results["VALID"].append((method, err_str))

        print(f"{method:<28s} {status:<22s} {diff_8_64:<14.6f} {diff_8_orig:<14.6f} {err_str:<22s}")

    print("\n" + "=" * 60)
    print("CATEGORICAL BREAKDOWN:")
    print("=" * 60)
    print(f"VALID RANK-SENSITIVE COMPRESSORS ({len(results['VALID'])}):")
    for m, e in results["VALID"]:
        print(f"  - {m:<28s} (Err: {e})")
    print(f"\nNO_OP (IGNORES RANK) ({len(results['NO_OP_IGNORES_RANK'])}):")
    for m, e in results["NO_OP_IGNORES_RANK"]:
        print(f"  - {m:<28s} (Err: {e})")
    print(f"\nNO_OP (FALLBACK TO ORIGINAL) ({len(results['NO_OP_FALLBACK'])}):")
    for m, e in results["NO_OP_FALLBACK"]:
        print(f"  - {m:<28s} (Err: {e})")
    print(f"\nFAILED / INCOMPATIBLE ({len(results['FAILED'])}):")
    for m, err in results["FAILED"]:
        print(f"  - {m:<28s} ({err})")

if __name__ == "__main__":
    run_diagnostic()
