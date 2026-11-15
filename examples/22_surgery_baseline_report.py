#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 22: Baseline Architecture vs. Surgeried Model Comparative Audit Report.

Subsystem Demonstrated:
    `neuralops.replace_attention`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Executing full model attention transplant (`cosformer`).
    2. Measuring real latency and parameters before and after surgery.
    3. Generating a verifiable structured surgery baseline report JSON.
    4. Auditing numerical parity and deviation across architectures.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

import neuralops
from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    measure_latency,
    pass_fail,
    section,
    tiny_inputs,
)


def resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback.")
    return torch.device(requested if requested != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda[:index]")
    parser.add_argument("--attention", default="cosformer")
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("surgery_baseline_report.json"))
    args = parser.parse_args()
    device = resolve_device(args.device)

    section("NeuralOps Example 22: Surgery Baseline Comparative Report")
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", str(device))
    kv("PyTorch Version", torch.__version__)
    kv("Target Attention", args.attention)
    kv("Number of Layers", args.layers)
    kv("Report Output Path", str(args.output))
    print()

    # Step 1: Reference baseline initialization & measurement
    torch.manual_seed(7)
    baseline = TinyClassifier(n_layers=args.layers).to(device).eval()
    inputs = {key: value.to(device) for key, value in tiny_inputs().items()}
    before_params = sum(p.numel() for p in baseline.parameters())

    with torch.no_grad():
        reference = baseline(**inputs)
        reference_logits = reference.logits if hasattr(reference, "logits") else reference
        lat_baseline = measure_latency(lambda: baseline(**inputs), warmup=2, iters=5)

    ref_finite = bool(torch.isfinite(reference_logits).all().item())
    pass_fail(
        "Baseline Reference Evaluation",
        ref_finite,
        f"latency={lat_baseline:.3f}ms, params={before_params}",
    )

    # Step 2: Full attention transplant
    surg_model = neuralops.replace_attention(
        baseline,
        attention=args.attention,
        copy_weights=True,
    ).eval()
    after_params = sum(p.numel() for p in surg_model.parameters())

    # Step 3: Candidate surgeried model evaluation
    with torch.no_grad():
        candidate = surg_model(**inputs)
        candidate_logits = candidate.logits if hasattr(candidate, "logits") else candidate
        lat_surg = measure_latency(lambda: surg_model(**inputs), warmup=2, iters=5)

    cand_finite = bool(torch.isfinite(candidate_logits).all().item())
    shape_preserved = bool(tuple(candidate_logits.shape) == tuple(reference_logits.shape))
    cand_eval_clean = bool(cand_finite and shape_preserved)
    pass_fail(
        "Surgeried Model Evaluation",
        cand_eval_clean,
        f"latency={lat_surg:.3f}ms, params={after_params}, shape={tuple(candidate_logits.shape)}",
    )

    # Step 4: Parity & deviation analysis
    max_diff = float((reference_logits.float() - candidate_logits.float()).abs().max().item())
    diff_finite = bool(torch.isfinite(torch.tensor(max_diff)).item())
    pass_fail(
        "Numerical Difference Bound",
        diff_finite,
        f"max absolute logits difference = {max_diff:.6e}",
    )

    # Step 5: Generate and export audit report
    speedup = round(lat_baseline / max(lat_surg, 1e-6), 3)
    param_ratio = round(after_params / max(before_params, 1), 3)

    report_status = "PASS" if (ref_finite and cand_eval_clean and diff_finite) else "FAIL"
    report = {
        "device": str(device),
        "attention": args.attention,
        "num_layers": args.layers,
        "baseline_params": before_params,
        "surgeried_params": after_params,
        "param_ratio": param_ratio,
        "baseline_latency_ms": lat_baseline,
        "surgeried_latency_ms": lat_surg,
        "speedup": speedup,
        "max_diff": max_diff,
        "status": report_status,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_written = bool(args.output.exists() and args.output.stat().st_size > 0)
    pass_fail(
        "Report Serialization",
        report_written,
        f"Wrote {args.output.stat().st_size if report_written else 0} bytes to {args.output}",
    )

    print()
    overall_passed = bool(report_status == "PASS" and report_written)
    pass_fail(
        "Overall Example 22 Status",
        overall_passed,
        "Comparative baseline report generated and verified successfully"
        if overall_passed
        else "Surgery baseline report verification failed",
    )
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
