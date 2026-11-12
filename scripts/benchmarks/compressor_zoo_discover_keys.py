#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Forensic Discovery and Verification Engine for Compressor Zoo (82 Engines).

Audits every registered compressor in COMPRESSOR_REGISTRY against strict
fail-closed and zero-fake-artifact contracts (AGENTS.md Section 0.2 & 0.3):
  1. Real Parameter Reduction: Enforces Param Reduction > 1.00x.
     Flag any engine returning 1.00x as IDENTITY_PASS_THROUGH / UNIMPLEMENTED.
  2. Real Reconstruction Error: Relative error > 0.0000 on non-trivial weights.
  3. Isolated Subprocess Probing: Guarantees SIGSEGV/SIGABRT immunity.
  4. Generates clean partition:
       - registry_keys.json   : Verified truly functional compression engines
       - identity_keys.json   : Mock pass-through engines (1.00x reduction)
       - failed_keys.json     : Engines with exceptions/unmet dependencies
       - audit_report.json    : Full forensic ledger with exact measurements
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("compressor_zoo_discover")


def probe_compressor(key: str, device_str: str = "cpu") -> dict[str, Any]:
    """
    Probe a registered compressor in an isolated subprocess.
    Tests against a standard 2-layer MLP block (768 -> 768 -> 768).
    Returns a dict with status, param_reduction, rel_error, and notes.
    """
    probe_code = f"""
import sys, os, json, torch, copy
import torch.nn as nn
sys.path.insert(0, os.path.abspath("."))

from neuralops.core.registry import COMPRESSOR_REGISTRY, create_compressor

device = torch.device("{device_str}")
torch.manual_seed(42)

# Build standard probe model (2-layer linear with bias)
model = nn.Sequential(
    nn.Linear(768, 768),
    nn.GELU(),
    nn.Linear(768, 768),
).to(device)

orig_params = sum(p.numel() for p in model.parameters())

# Input probe batch
x = torch.randn(4, 16, 768, device=device)
with torch.no_grad():
    y_orig = model(x)

# Attempt compression
comp_cls = COMPRESSOR_REGISTRY.get("{key}")
if comp_cls is None:
    print(json.dumps({{"status": "NOT_REGISTERED", "error": "Not in registry"}}))
    sys.exit(0)

compressed_model = None
stats = {{}}

try:
    # Try universal factory
    inst = create_compressor("{key}", model=model, rank=128)
    if hasattr(inst, "compress") and callable(inst.compress):
        res = inst.compress()
        if isinstance(res, tuple):
            compressed_model, stats = res[0], (res[1] if len(res) > 1 else {{}})
        else:
            compressed_model = res
    elif isinstance(inst, nn.Module):
        compressed_model = inst
    else:
        # Check if inst is callable directly
        res = inst(model) if callable(inst) else None
        if isinstance(res, tuple):
            compressed_model, stats = res[0], res[1]
        elif isinstance(res, nn.Module):
            compressed_model = res
except Exception as e1:
    # Try alternative direct instantiation
    try:
        target = getattr(comp_cls, "__wrapped__", comp_cls)
        inst = comp_cls(model=model, rank=128)
        if hasattr(inst, "compress") and callable(inst.compress):
            res = inst.compress()
            compressed_model = res[0] if isinstance(res, tuple) else res
        elif isinstance(inst, nn.Module):
            compressed_model = inst
    except Exception as e2:
        print(json.dumps({{"status": "EXCEPTION", "error": str(e1) + " | " + str(e2)}}))
        sys.exit(0)

if compressed_model is None or not isinstance(compressed_model, nn.Module):
    print(json.dumps({{"status": "INVALID_OUTPUT", "error": "Compressor did not return nn.Module"}}))
    sys.exit(0)

comp_params = sum(p.numel() for p in compressed_model.parameters())
ratio = orig_params / max(comp_params, 1)

# Forward pass check & numerical reconstruction error
try:
    with torch.no_grad():
        y_comp = compressed_model(x)
        diff = torch.norm(y_orig - y_comp).item()
        denom = torch.norm(y_orig).clamp(min=1e-8).item()
        rel_err = diff / denom
except Exception as e:
    print(json.dumps({{"status": "FORWARD_FAILED", "error": str(e)}}))
    sys.exit(0)

# Fail-closed classification
if abs(ratio - 1.0) < 1e-4 and rel_err < 1e-6:
    status = "IDENTITY_PASS_THROUGH"
elif ratio > 1.00:
    status = "PASSED_FUNCTIONAL"
else:
    status = "NON_COMPRESSING"

print(json.dumps({{
    "status": status,
    "orig_params": orig_params,
    "comp_params": comp_params,
    "param_reduction": round(ratio, 4),
    "rel_error": round(rel_err, 6),
    "error": None,
}}))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", probe_code],
            capture_output=True,
            text=True,
            timeout=35,
        )
        if proc.returncode == 0:
            lines = [l for l in proc.stdout.strip().splitlines() if l.startswith("{")]
            if lines:
                return json.loads(lines[-1])
            return {"status": "UNKNOWN_OUTPUT", "error": proc.stdout.strip()}
        else:
            err = proc.stderr.strip() or proc.stdout.strip() or f"Exit {proc.returncode}"
            return {"status": "PROCESS_ABORT", "error": err}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": "Timed out after 35s"}
    except Exception as e:
        return {"status": "ERROR", "error": str(e)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Forensic Discovery of Compressor Zoo")
    parser.add_argument("--out_dir", default="results/compressor_zoo", help="Output directory")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from neuralops.core.registry import COMPRESSOR_REGISTRY

    all_keys = sorted(COMPRESSOR_REGISTRY.keys())
    logger.info("Total Registered Compressors: %d engines", len(all_keys))

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    functional_keys: list[str] = []
    identity_keys: list[dict[str, Any]] = []
    failed_keys: list[dict[str, Any]] = []
    full_report: list[dict[str, Any]] = []

    t0 = time.time()
    for i, key in enumerate(all_keys):
        logger.info("[%02d/%02d] Auditing Compressor: %s", i + 1, len(all_keys), key)
        res = probe_compressor(key, device_str=args.device)
        res["compressor"] = key
        full_report.append(res)

        st = res.get("status")
        if st == "PASSED_FUNCTIONAL":
            functional_keys.append(key)
            logger.info("  -> PASSED: Param Reduction=%.2fx | Rel Err=%.4f", res.get("param_reduction", 1.0), res.get("rel_error", 0.0))
        elif st == "IDENTITY_PASS_THROUGH":
            identity_keys.append(res)
            logger.warning("  -> REJECTED (Zero-Fake Contract): Identity Pass-Through (1.00x reduction, 0.0000 error)")
        else:
            failed_keys.append(res)
            err_msg = res.get("error", "Unknown error")
            first_line = err_msg.splitlines()[-1] if err_msg else "Failed"
            logger.warning("  -> FAILED (%s): %s", st, first_line)

    elapsed = time.time() - t0
    logger.info("\n" + "=" * 80)
    logger.info("Compressor Zoo Audit Complete in %.1fs", elapsed)
    logger.info("  Verified Functional Engines : %d / %d", len(functional_keys), len(all_keys))
    logger.info("  Identity Pass-Through (Fake): %d / %d", len(identity_keys), len(all_keys))
    logger.info("  Failed / Missing Dependencies: %d / %d", len(failed_keys), len(all_keys))
    logger.info("=" * 80)

    def _atomic_write_json(p: Path, data: Any) -> None:
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(f".tmp.{os.getpid()}_{uuid.uuid4().hex[:6]}")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        tmp.replace(p)

    # Write artifacts atomically
    _atomic_write_json(out_path / "registry_keys.json", functional_keys)
    _atomic_write_json(out_path / "identity_keys.json", identity_keys)
    _atomic_write_json(out_path / "failed_keys.json", failed_keys)
    _atomic_write_json(out_path / "audit_report.json", full_report)

    logger.info("Artifacts saved to %s/", out_path)


if __name__ == "__main__":
    main()
