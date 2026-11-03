#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Compare a model before and after full attention surgery."""

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
from examples._shared import TinyClassifier, tiny_inputs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--attention", default="cosformer")
    parser.add_argument("--output", type=Path, default=Path("surgery_baseline_report.json"))
    args = parser.parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback.")
    device = torch.device(args.device)

    torch.manual_seed(7)
    baseline = TinyClassifier(n_layers=3).to(device).eval()
    inputs = {key: value.to(device) for key, value in tiny_inputs().items()}
    with torch.no_grad():
        reference = baseline(**inputs)
        reference_logits = reference.logits if hasattr(reference, "logits") else reference

    surg_model = neuralops.replace_attention(baseline, attention=args.attention, copy_weights=True).eval()
    with torch.no_grad():
        candidate = surg_model(**inputs)
        candidate_logits = candidate.logits if hasattr(candidate, "logits") else candidate
    max_diff = float((reference_logits.float() - candidate_logits.float()).abs().max().item())
    report = {
        "device": str(device),
        "attention": args.attention,
        "max_diff": max_diff,
        "status": "PASS" if torch.isfinite(candidate_logits).all().item() else "FAIL_NONFINITE",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        raise RuntimeError(f"Surgery produced non-finite output: {report}")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
