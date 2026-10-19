#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Forensic diagnostic script to verify rank-sensitivity across all 26 compressors.
Classifies each compressor into:
  1. VALID: rank changes the compressed weights and error (legitimate low-rank SVD / rank-sensitive)
  2. NO_OP_FALLBACK: rejected compression and fell back to original weights (error=0.0)
  3. NO_OP_IGNORES_RANK: compressed weights at rank=8 and rank=64 are identical (ignores rank parameter)
  4. FAILED: crashes or constructor fails
"""

import os
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
)


def extract_reconstructed_weights(model: nn.Module) -> torch.Tensor:
    """Extract reconstructed full-rank dense weights across layers."""
    tensors = []
    for name, m in model.named_modules():
        if hasattr(m, "U") and hasattr(m, "V") and m.U is not None and m.V is not None:
            U, V = m.U.data.float(), m.V.data.float()
            if U.shape[1] == V.shape[0]:
                w = torch.matmul(U, V)
            elif U.shape[1] == V.shape[1]:
                w = torch.matmul(U, V.t())
            elif U.shape[0] == V.shape[0]:
                w = torch.matmul(U.t(), V)
            else:
                w = torch.matmul(U, V)
            tensors.append(w.reshape(-1))
        elif hasattr(m, "weight") and m.weight is not None and m.weight.ndim == 2 and m.weight.numel() > 100:
            tensors.append(m.weight.data.float().reshape(-1))
    if tensors:
        return torch.cat(tensors)
    return torch.tensor([0.0])


def run_diagnostic(model_name: str = "synth_block"):
    """Execute run diagnostic routine."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Running Rank Sensitivity Diagnostic on model '%s' via %s", model_name, device)

    loaders = {
        "synth_block": load_synthetic_block,
        "roberta": load_roberta,
        "gpt2": load_gpt2,
    }
    torch.manual_seed(42)
    anchor = loaders[model_name](device)
    probe, heldout, _, _ = get_probe_data(model_name, device)

    with torch.no_grad():
        if model_name == "roberta":
            ids, mask = heldout
            y_anchor = anchor(ids, attention_mask=mask).logits
        elif model_name == "gpt2":
            y_anchor = anchor(heldout).logits
        else:
            y_anchor = anchor(heldout)

    results = {
        "VALID": [],
        "NO_OP_IGNORES_RANK": [],
        "NO_OP_FALLBACK": [],
        "FAILED": [],
    }

    print("\n" + "=" * 115)
    print(f"RANK SENSITIVITY DIAGNOSTIC: {model_name.upper()}")
    print("=" * 115)
    print(
        f"{'Method':<28s} {'Status':<22s} {'LogitDiff(8,64)':<16s} {'RelErr(r8)':<12s} {'RelErr(r64)':<12s} {'Details'}"
    )
    print("-" * 115)

    for method in COMPRESSOR_REGISTRY_KEYS:
        m8 = copy.deepcopy(anchor)
        comp8 = get_compressor(method, m8, 8, probe)
        if comp8 is None:
            results["FAILED"].append((method, "Constructor returned None"))
            print(f"{method:<28s} {'FAILED (None)':<22s} {'—':<16s} {'—':<12s} {'—':<12s} Constructor failed")
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

            with torch.no_grad():
                if model_name == "roberta":
                    ids, mask = heldout
                    out8_logits = m8(ids, attention_mask=mask).logits
                elif model_name == "gpt2":
                    out8_logits = m8(heldout).logits
                else:
                    out8_logits = m8(heldout)
            err8 = (torch.norm(out8_logits.float() - y_anchor.float()) / torch.norm(y_anchor.float())).item()
        except Exception as e:
            results["FAILED"].append((method, str(e)[:40]))
            print(f"{method:<28s} {'FAILED (Exec r8)':<22s} {'—':<16s} {'—':<12s} {'—':<12s} {str(e)[:30]}")
            continue

        m64 = copy.deepcopy(anchor)
        comp64 = get_compressor(method, m64, 64, probe)
        if comp64 is None:
            results["FAILED"].append((method, "Rank 64 constructor failed"))
            print(f"{method:<28s} {'FAILED (r64 None)':<22s} {'—':<16s} {'—':<12s} {'—':<12s} Rank 64 init failed")
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

            with torch.no_grad():
                if model_name == "roberta":
                    ids, mask = heldout
                    out64_logits = m64(ids, attention_mask=mask).logits
                elif model_name == "gpt2":
                    out64_logits = m64(heldout).logits
                else:
                    out64_logits = m64(heldout)
            err64 = (torch.norm(out64_logits.float() - y_anchor.float()) / torch.norm(y_anchor.float())).item()
        except Exception as e:
            results["FAILED"].append((method, str(e)[:40]))
            print(f"{method:<28s} {'FAILED (Exec r64)':<22s} {'—':<16s} {'—':<12s} {'—':<12s} {str(e)[:30]}")
            continue

        logit_diff = (out8_logits - out64_logits).abs().max().item()
        err_diff = abs(err8 - err64)
        err_str = f"{err8:.4f} -> {err64:.4f}"

        if err8 < 1e-6 and err64 < 1e-6:
            status = "NO_OP_FALLBACK"
            results["NO_OP_FALLBACK"].append((method, err_str))
            detail = "Identity/Fallback to uncompressed (0 error)"
        elif logit_diff < 1e-5 or err_diff < 1e-4:
            status = "NO_OP_IGNORES_RANK"
            results["NO_OP_IGNORES_RANK"].append((method, err_str))
            detail = "Constant output across ranks (rank ignored)"
        else:
            status = "VALID"
            results["VALID"].append((method, err_str))
            detail = f"Rank-sensitive: error drops by {err8 - err64:+.4f}"

        print(f"{method:<28s} {status:<22s} {logit_diff:<16.6f} {err8:<12.4f} {err64:<12.4f} {detail}")

    print("\n" + "=" * 60)
    print(f"CATEGORICAL BREAKDOWN ({model_name.upper()}):")
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

    out_dir = Path("results/batch_validation") / "diagnostics"
    out_dir.mkdir(parents=True, exist_ok=True)
    import json

    with open(out_dir / f"{model_name}_diagnostic.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="synth_block", choices=["synth_block", "roberta", "gpt2"])
    args = parser.parse_args()
    run_diagnostic(args.model)
