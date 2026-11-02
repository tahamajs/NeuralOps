#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 01: Attention Surgery across 5 Architectural Families and Hybrid Layering.

Subsystem Demonstrated:
    `neuralops.replace_attention`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Replacing quadratic self-attention across 5 representative families:
       - Kernelized linear: Cosformer
       - Recurrent linear: Gated Linear Attention (GLA)
       - Chunk/Decay linear: Decay Linear Attention
       - State-Space model: Mamba block
       - Low-Rank SVD: SVD Linear Attention
    2. Constructing hybrid architectures by selective layer replacement (`layers=[0]`).
    3. Controlling weight transplantation dynamics (`copy_weights=True` vs `copy_weights=False`).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure repo root is on sys.path for examples._shared resolution
ROOT = Path(__file__).resolve().parent.parent
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


def demonstrate_five_families(device: str) -> bool:
    """Test attention surgery across 5 distinct architectural families."""
    section("Part 1: 5 Architectural Families Demonstration")

    families = [
        ("Kernelized Linear", "cosformer"),
        ("Recurrent Linear", "gla_attention"),
        ("Chunk/Decay Linear", "decay_linear_attention"),
        ("State-Space Model", "mamba_block"),
        ("Low-Rank SVD Attention", "svd_linear_attention"),
    ]

    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}
    all_passed = True

    for family_name, attn_name in families:
        base_model = TinyClassifier().to(device)
        params_before = base_model.count_parameters()

        # Perform in-place attention replacement
        surg_model = neuralops.replace_attention(
            base_model,
            attention=attn_name,
            copy_weights=False,
        )
        params_after = sum(p.numel() for p in surg_model.parameters())

        with torch.no_grad():
            output = surg_model(**inputs)
            logits = output.logits if hasattr(output, "logits") else output

        is_finite = bool(torch.isfinite(logits).all().item())
        all_passed = all_passed and is_finite

        kv(f"{family_name} ({attn_name})", f"{params_before} -> {params_after} params")
        pass_fail(f"{attn_name} forward pass", is_finite, f"logits shape={tuple(logits.shape)}")

    return all_passed


def demonstrate_hybrid_layers(device: str) -> bool:
    """Test layer-specific surgery: replacing even layers, keeping odd layers intact."""
    section("Part 2: Layer-Specific Hybrid Architecture")

    model = TinyClassifier(n_layers=2).to(device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}

    # Replace only layer 0 (even layer) with GLA, leaving layer 1 as full attention
    hybrid_model = neuralops.replace_attention(
        model,
        attention="gla_attention",
        layers=[0],
        copy_weights=False,
    )

    with torch.no_grad():
        out = hybrid_model(**inputs)
        logits = out.logits if hasattr(out, "logits") else out

    is_finite = bool(torch.isfinite(logits).all().item())
    pass_fail("Hybrid Model (Layer 0 GLA + Layer 1 Full Attention)", is_finite)
    return is_finite


def demonstrate_weight_transplantation(device: str) -> bool:
    """Verify that copy_weights=True preserves projected weights while False initializes afresh."""
    section("Part 3: Weight Transplantation Modes")

    m_copied = TinyClassifier(seed=100).to(device)
    m_fresh = TinyClassifier(seed=100).to(device)

    neuralops.replace_attention(m_copied, attention="cosformer", copy_weights=True)
    neuralops.replace_attention(m_fresh, attention="cosformer", copy_weights=False)

    # Compare parameter tensors between copied and freshly initialized models
    params_copied = torch.cat([p.flatten() for p in m_copied.parameters()])
    params_fresh = torch.cat([p.flatten() for p in m_fresh.parameters()])

    weights_differ = not torch.equal(params_copied, params_fresh)
    pass_fail(
        "Weight Transplantation Difference (copy_weights=True vs False)",
        weights_differ,
        "Confirmed state divergence between copied and fresh initialization",
    )
    return weights_differ


def main() -> int:
    section("NeuralOps Example 01: Comprehensive Attention Surgery")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    p1 = demonstrate_five_families(backend)
    p2 = demonstrate_hybrid_layers(backend)
    p3 = demonstrate_weight_transplantation(backend)

    success = p1 and p2 and p3
    print()
    if success:
        pass_fail("Overall Example 01 Status", True, "All surgery and hybrid checks passed")
        return 0
    else:
        pass_fail("Overall Example 01 Status", False, "One or more checks failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
