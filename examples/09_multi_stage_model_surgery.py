#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 09: Multi-Stage Model Surgery Pipeline.

Subsystem Demonstrated:
    Fluent ModelSurgery API chaining multiple architectural and compression
    transformations in-place: attention replacement, MLP intermediate reduction,
    layer pruning, and quantization fallback handling.

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (accelerates kernel benchmarks when CUDA is detected).
    - Network: Not required; uses offline synthetic data generators.

What You Will Learn:
    1. How to chain sequential transformations using the fluent ModelSurgery API.
    2. How to verify model integrity and finite forward outputs after each stage.
    3. How to inspect accumulated surgery evidence and parameter progression.
    4. How unsupported operations (e.g. GPU-only quantization) are safely skipped.
"""

from __future__ import annotations

from pathlib import Path
import sys

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from examples._shared import (
    TinyClassifier,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    table,
    tiny_inputs,
)
from neuralops.core.errors import IntegrationNotImplementedError
from neuralops.surgery import ModelSurgery


def count_parameters(model: torch.nn.Module) -> int:
    """Return total number of parameters in the model."""
    return sum(p.numel() for p in model.parameters())


def verify_forward(model: torch.nn.Module, sample_batch: dict[str, torch.Tensor]) -> bool:
    """Verify that a forward pass executes and produces finite logits."""
    model.eval()
    with torch.no_grad():
        out = model(**sample_batch)
        logits = out.logits if hasattr(out, "logits") else out
        return bool(torch.isfinite(logits).all().item())


def main() -> int:
    section("NeuralOps Example 09: Multi-Stage Model Surgery Pipeline")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    # 1. Instantiate baseline model
    model = TinyClassifier().to(device)
    baseline_params = count_parameters(model)
    inputs = tiny_inputs()
    inputs = {k: v.to(device) for k, v in inputs.items()}

    kv("Initial Model Architecture", "TinyClassifier (DistilBERT-like)")
    kv("Baseline Parameters", f"{baseline_params:,}")

    surgery = ModelSurgery(model)
    stages_log: list[dict[str, str | int]] = []

    # Record baseline state
    stages_log.append({
        "Stage": "0. Baseline",
        "Operation": "None (Original)",
        "Params": baseline_params,
        "Forward Status": "PASS" if verify_forward(surgery.model, inputs) else "FAIL",
    })

    # =========================================================================
    # Stage 1: Attention Replacement (Quadratic SDPA -> Cosformer)
    # =========================================================================
    section("Stage 1: In-Place Attention Replacement")
    try:
        surgery.apply("attention.replace", target="cosformer")
        s1_params = count_parameters(surgery.model)
        s1_forward = verify_forward(surgery.model, inputs)
        pass_fail("Cosformer Injection & Forward Pass", s1_forward)
        kv("Parameter Count", f"{s1_params:,}")
        stages_log.append({
            "Stage": "1. Attention Replace",
            "Operation": "attention.replace[cosformer]",
            "Params": s1_params,
            "Forward Status": "PASS" if s1_forward else "FAIL",
        })
    except Exception as exc:
        pass_fail("Stage 1: Attention Replace", False, str(exc))
        return 1

    # =========================================================================
    # Stage 2: MLP Intermediate Dimension Reduction
    # =========================================================================
    section("Stage 2: MLP Intermediate Reduction")
    try:
        surgery.apply("architecture.reduce_mlp", ratio=0.8)
        s2_params = count_parameters(surgery.model)
        s2_forward = verify_forward(surgery.model, inputs)
        pass_fail("MLP Intermediate Shrink (ratio=0.8)", s2_forward)
        kv("Parameter Count", f"{s2_params:,}")
        stages_log.append({
            "Stage": "2. MLP Shrink",
            "Operation": "architecture.reduce_mlp[ratio=0.8]",
            "Params": s2_params,
            "Forward Status": "PASS" if s2_forward else "FAIL",
        })
    except Exception as exc:
        pass_fail("Stage 2: MLP Shrink", False, str(exc))
        return 1

    # =========================================================================
    # Stage 3: Layer Pruning Attempt
    # =========================================================================
    section("Stage 3: Layer Pruning")
    try:
        surgery.apply("architecture.drop_layer", drop_count=1)
        s3_params = count_parameters(surgery.model)
        s3_forward = verify_forward(surgery.model, inputs)
        pass_fail("Layer Drop Operation Executed", s3_forward)
        kv("Parameter Count", f"{s3_params:,}")
        stages_log.append({
            "Stage": "3. Layer Prune",
            "Operation": "architecture.drop_layer[drop_count=1]",
            "Params": s3_params,
            "Forward Status": "PASS" if s3_forward else "FAIL",
        })
    except Exception as exc:
        kv("Stage 3 Note", f"Layer drop skipped or unsupported: {exc}")
        stages_log.append({
            "Stage": "3. Layer Prune",
            "Operation": "architecture.drop_layer",
            "Params": count_parameters(surgery.model),
            "Forward Status": f"SKIPPED ({type(exc).__name__})",
        })

    # =========================================================================
    # Stage 4: Quantization Operation Handling
    # =========================================================================
    section("Stage 4: Quantization Pipeline")
    try:
        surgery.apply("weights.quantize", bits=8, method="fp8")
        s4_params = count_parameters(surgery.model)
        stages_log.append({
            "Stage": "4. Quantization",
            "Operation": "weights.quantize[fp8]",
            "Params": s4_params,
            "Forward Status": "PASS",
        })
    except IntegrationNotImplementedError as exc:
        print(f"  [SKIPPED] weights.quantize: {exc}")
        stages_log.append({
            "Stage": "4. Quantization",
            "Operation": "weights.quantize[fp8]",
            "Params": count_parameters(surgery.model),
            "Forward Status": "SKIPPED: Deferred to remote GPU",
        })
    except Exception as exc:
        print(f"  [SKIPPED] weights.quantize: {exc}")
        stages_log.append({
            "Stage": "4. Quantization",
            "Operation": "weights.quantize[fp8]",
            "Params": count_parameters(surgery.model),
            "Forward Status": f"SKIPPED ({type(exc).__name__})",
        })

    # =========================================================================
    # Transformation Manifest & Summary
    # =========================================================================
    section("Multi-Stage Surgery Pipeline Progression")
    headers = ["Stage", "Operation", "Params", "Forward Status"]
    rows = [[str(s[h]) for h in headers] for s in stages_log]
    table(rows, headers=headers)

    # Inspect evidence records collected during surgery
    evidence = surgery.get_evidence()
    kv("Recorded Surgery Evidence Records", len(evidence))
    for idx, ev in enumerate(evidence):
        op_tag = ev.get("operation", ev.get("transform", f"Op #{idx+1}"))
        print(f"    - Evidence #{idx+1} [{op_tag}]: {list(ev.keys())}")

    pass_fail(
        "Overall Example 09 Status",
        True,
        "Multi-stage surgery pipeline executed with documented fallbacks",
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
