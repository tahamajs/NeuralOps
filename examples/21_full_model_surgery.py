#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 21: Full-Model Fail-Closed Attention Surgery across all Transformer Blocks.

Subsystem Demonstrated:
    `neuralops.replace_attention`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Executing full-model attention transplants across every transformer block (`replace_attention`).
    2. Dynamically verifying post-transplant layer architecture invariants.
    3. Performing backward pass certification verifying finite gradient propagation to all parameters.
    4. Asserting numerical finiteness, shape preservation, and real surgery integrity metrics.
"""

from __future__ import annotations

import argparse
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
    args = parser.parse_args()
    device = resolve_device(args.device)

    section("NeuralOps Example 21: Full-Model Surgery")
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", str(device))
    kv("PyTorch Version", torch.__version__)
    kv("Requested Attention", args.attention)
    kv("Target Layers Count", args.layers)
    print()

    # Step 1: Initialize baseline model and inputs
    torch.manual_seed(42)
    model = TinyClassifier(n_layers=args.layers).to(device)
    inputs = {key: value.to(device) for key, value in tiny_inputs().items()}
    before_params = sum(parameter.numel() for parameter in model.parameters())

    # Step 2: Execute full attention surgery across every block
    surg_model = neuralops.replace_attention(
        model,
        attention=args.attention,
        copy_weights=True,
    )
    after_params = sum(parameter.numel() for parameter in surg_model.parameters())

    # Step 3: Layer replacement verification
    replaced_count = sum(
        1
        for mod in surg_model.modules()
        if mod.__class__.__name__ == "UniversalAttentionWrapper"
    )
    layers_replaced_clean = bool(replaced_count == args.layers)
    pass_fail(
        "All Layers Attention Replacement",
        layers_replaced_clean,
        f"{replaced_count}/{args.layers} transformer layers injected with {args.attention}",
    )

    # Step 4: Forward pass verification
    output = surg_model(**inputs)
    logits = output.logits if hasattr(output, "logits") else output
    finite_output = bool(torch.isfinite(logits).all().item())
    expected_shape = (inputs["input_ids"].shape[0], 2)
    shape_preserved = bool(tuple(logits.shape) == expected_shape)
    forward_clean = bool(finite_output and shape_preserved)
    pass_fail(
        "Full-Model Forward Pass Integrity",
        forward_clean,
        f"logits shape={tuple(logits.shape)}, all finite={finite_output}",
    )

    # Step 5: Backward pass & gradient certification
    loss = logits.float().square().mean()
    loss.backward()

    gradients = [parameter.grad for parameter in surg_model.parameters() if parameter.requires_grad]
    finite_gradients = bool(gradients) and all(
        gradient is not None and bool(torch.isfinite(gradient).all().item()) for gradient in gradients
    )
    pass_fail(
        "Trainable Gradients Propagation",
        finite_gradients,
        f"{len(gradients)}/{len(gradients)} active gradients present and numerically finite",
    )

    # Step 6: Parameter delta logging
    kv("Parameter Count Delta", f"{before_params} -> {after_params}")

    overall_passed = bool(layers_replaced_clean and forward_clean and finite_gradients)
    print()
    pass_fail(
        "Overall Example 21 Status",
        overall_passed,
        "All full-model surgery checks passed" if overall_passed else "One or more surgery checks failed",
    )
    return 0 if overall_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
