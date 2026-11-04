#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 23: Universal Surgery Director & The Ten Inviolable Laws.

Subsystem Demonstrated:
    Universal Surgery Director invariant compliance engine across canonical model families:
    1. Pre-Feature-Map QK-Norm Preservation Hook (Law 1)
    2. Logit Soft-Capping Temperature Folding (Law 2)
    3. Zero-Bleed Padding Mask Sanitization (Law 3)
    4. Static Cross-Attention Associative Caching (Law 4)
    5. RoPE Pre-Projection Alignment (Law 5)
    6. Decoupled Latent State Router for DeepSeek MLA (Law 6)
    7. Causal State Monotonicity & Exponential State Decay (Law 7)
    8. Tied Embedding Preservation & Auto-Remediation (Law 8)
    9. FP32 Accumulation & Dtype Budget Guard (Law 9)
    10. Master Universal Surgery Integration & Certification (Law 10)

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional (accelerates kernel benchmarks when CUDA is detected).
    - Network: Not required; uses offline synthetic data generators.

What You Will Learn:
    1. How UniversalSurgeryDirector profiles model architectures into invariant contracts.
    2. How the Ten Inviolable Laws protect mathematical soundness during attention transplants.
    3. How to execute automated invariant certification and render structured compliance reports.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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
from neuralops.surgery.director import UniversalSurgeryDirector


def resolve_device(requested: str) -> torch.device:
    """Resolve compute device with strict fail-closed policy on CUDA availability."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; refusing CPU fallback.")
    if requested == "auto":
        pref = get_backend_preference()
        return torch.device(pref)
    return torch.device(requested)


def run_demonstration(device: torch.device, target_attention: str = "cosformer") -> bool:
    """Execute Universal Surgery Director demonstration and verify Ten Laws compliance."""
    # 1. Initialize deterministic base classifier
    model = TinyClassifier(n_layers=2).to(device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}

    # Compute baseline reference forward pass
    with torch.no_grad():
        ref_out = model(**inputs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out

    # 2. Instantiate UniversalSurgeryDirector and inspect architectural profile
    director = UniversalSurgeryDirector(model)
    profile = director.profile

    kv("Detected Model Family", profile.family)
    kv("Hidden Dimension", profile.hidden_dim)
    kv("Number of Layers", profile.num_layers)
    kv("Has Tied Embeddings", profile.has_tied_embeddings)

    # 3. Execute Attention Surgery enforcing all Ten Inviolable Laws
    surg_model = director.execute_surgery(
        target_attention=target_attention,
        inherit_weights=True,
    )

    # 4. Verify post-surgery forward pass
    with torch.no_grad():
        surg_out = surg_model(**inputs)
        surg_logits = surg_out.logits if hasattr(surg_out, "logits") else surg_out

    is_finite = bool(torch.isfinite(surg_logits).all().item())
    shape_matches = tuple(surg_logits.shape) == tuple(ref_logits.shape)
    pass_fail("Forward Output Finite Check", is_finite)
    pass_fail("Logit Shape Parity Check", shape_matches, f"{tuple(surg_logits.shape)}")

    # 5. Run Invariant Certification Engine across the Ten Laws
    cert_report = director.certify_model(strict=True)
    laws_status = cert_report.get("laws_certified", {})

    # Display structured table of the Ten Inviolable Laws
    law_descriptions = {
        "Law_1_QK_Norm": "Pre-Feature-Map QK-Norm Preservation",
        "Law_2_Softcapping": "Logit Soft-Capping Temperature Folding",
        "Law_3_Zero_Bleed_Padding": "Zero-Bleed Padding Mask Sanitization",
        "Law_4_Static_Cross_Attention": "Static Cross-Attention Associative Caching",
        "Law_5_RoPE_Pre_Projection": "RoPE Pre-Projection Alignment",
        "Law_6_Decoupled_Latent_Router": "Decoupled Latent State Router (DeepSeek MLA)",
        "Law_7_Causal_Monotonicity_State_Decay": "Causal State Monotonicity & Exponential Decay",
        "Law_8_Tied_Embeddings": "Tied Embedding Preservation & Auto-Remediation",
        "Law_9_FP32_Accumulation": "FP32 Accumulation & Dtype Budget Guard",
        "Law_10_Master_Integration": "Master Universal Surgery Integration",
    }

    headers = ["Law Code", "Description", "Certified"]
    report_rows: list[list[str]] = []
    for law_key, description in law_descriptions.items():
        status = laws_status.get(law_key, False)
        report_rows.append([
            law_key.replace("_", " "),
            description,
            "PASSED" if status else "FAILED",
        ])

    print()
    table(report_rows, headers=headers)

    is_certified = bool(cert_report.get("certified", False))
    pass_fail("Ten Laws Certification Score", is_certified, cert_report.get("score", "N/A"))

    return bool(is_finite and shape_matches and is_certified)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="auto", help="Execution device: auto, cpu, or cuda[:index]")
    parser.add_argument("--attention", default="cosformer", help="Target linear attention operator")
    args = parser.parse_args()

    device = resolve_device(args.device)

    section("NeuralOps Example 23: Universal Surgery Director & The Ten Inviolable Laws")
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", str(device))
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_demonstration(device=device, target_attention=args.attention)

    print()
    if success:
        pass_fail("Overall Example 23 Status", True, "All Ten Laws certified and forward verified")
        return 0
    else:
        pass_fail("Overall Example 23 Status", False, "Surgery certification or verification failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
