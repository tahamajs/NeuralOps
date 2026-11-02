#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example Template: Canonical Skeleton for NeuralOps Examples.

Subsystem Demonstrated:
    [Name of Subsystem, e.g., Model Surgery, SVD Compression, Distillation]

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional (accelerates kernel benchmarks when CUDA is detected).
    - Network: Not required; uses offline synthetic data generators.

What You Will Learn:
    1. How to initialize and configure [Target Subsystem/API].
    2. How to apply architectural transformations or compression in-place.
    3. How to verify numerical correctness and parameter invariants.
    4. How to measure execution latency and memory footprint.
"""

from __future__ import annotations

import sys
import torch

# 1. Import shared infrastructure from examples._shared
from examples._shared import (
    TinyClassifier,
    baseline_output,
    get_backend_preference,
    gpu_name,
    kv,
    pass_fail,
    section,
    tiny_inputs,
)

# 2. Import target NeuralOps subsystems
import neuralops


def run_demonstration(device: str) -> bool:
    """
    Execute the core subsystem demonstration steps.

    Args:
        device: Device string ('cpu', 'cuda', or 'mps').

    Returns:
        True if all demonstration steps and parity checks pass, False otherwise.
    """
    # Step A: Initialize model and reference inputs
    model = TinyClassifier().to(device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}
    ref_output = baseline_output(model, inputs)

    # Step B: Perform subsystem operation (e.g., surgery, compression, distillation)
    # [Insert specific NeuralOps API call here, e.g.:]
    # modified_model = neuralops.replace_attention(model, attention="cosformer")
    modified_model = model  # Placeholder for template

    # Step C: Execute forward pass on transformed model
    with torch.no_grad():
        out = modified_model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out

    # Step D: Verify numerical validity
    is_finite = torch.isfinite(logits).all().item()
    pass_fail("Logits Finite Check", is_finite)

    # Step E: Verify expected output shape
    shape_matches = tuple(logits.shape) == tuple(ref_output.shape)
    pass_fail("Output Shape Parity", shape_matches, f"{tuple(logits.shape)}")

    return bool(is_finite and shape_matches)


def main() -> int:
    """
    Main entry point for the example.

    Returns:
        0 on success, 1 on failure.
    """
    # 1. Print boxed section header
    section("NeuralOps Example Template")

    # 2. Report hardware and execution environment
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Preferred Device", backend)
    kv("PyTorch Version", torch.__version__)
    kv("NeuralOps Version", getattr(neuralops, "__version__", "unknown"))
    print()

    # 3. Run demonstration steps and verification
    success = run_demonstration(device=backend)

    # 4. Print summary status
    print()
    if success:
        pass_fail("Overall Example Status", True, "All verification steps passed")
        return 0
    else:
        pass_fail("Overall Example Status", False, "One or more verification steps failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
