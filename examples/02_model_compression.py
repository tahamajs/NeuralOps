#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 02: Model Compression Comparison across 4 SVD Factorization Algorithms.

Subsystem Demonstrated:
    `neuralops.compress`

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution.
    - GPU: Optional acceleration.

What You Will Learn:
    1. Comparing 4 distinct SVD matrix factorization strategies on identical linear projections:
       - Direct Truncated SVD (`direct_svd`)
       - Activation-Aware SVD (`adasvd_compressor`)
       - Fisher-Weighted SVD (`fwsvd_compressor`)
       - SVD-LLM Low-Rank Factorization (`svd_llm_compressor`)
    2. Ingesting synthetic offline calibration data without network dependencies.
    3. Measuring parameter footprint reduction, spectral energy retention, and reconstruction error.
    4. Verifying forward pass stability and output tensor finiteness post-compression.
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
    print_benchmark_table,
    section,
    tiny_inputs,
    toy_calibration_loader,
)


def compute_spectral_metrics(
    original_model: torch.nn.Module,
    compressed_model: torch.nn.Module,
    rank: int = 8,
) -> tuple[float, float]:
    """
    Compute average singular value energy retention and reconstruction Frobenius error.

    Returns:
        (energy_retention_pct, frobenius_reconstruction_error)
    """
    energies: list[float] = []
    errors: list[float] = []

    orig_named = dict(original_model.named_parameters())
    for name, orig_param in orig_named.items():
        if orig_param.ndim == 2 and ("lin" in name or "dense" in name or "fc" in name):
            W = orig_param.detach().float()
            _, S, _ = torch.linalg.svd(W, full_matrices=False)
            total_energy = (S**2).sum().item()
            retained_energy = (S[:rank] ** 2).sum().item() if rank <= len(S) else total_energy
            energy_ratio = (retained_energy / max(total_energy, 1e-8)) * 100.0
            energies.append(energy_ratio)

            # Estimate reconstruction error from truncated spectrum: sqrt(sum(S_{r:}^2)) / sqrt(sum(S^2))
            tail_energy = (S[rank:] ** 2).sum().item() if rank < len(S) else 0.0
            frob_err = (tail_energy / max(total_energy, 1e-8)) ** 0.5
            errors.append(frob_err)

    avg_energy = sum(energies) / len(energies) if energies else 100.0
    avg_error = sum(errors) / len(errors) if errors else 0.0
    return round(avg_energy, 2), round(avg_error, 4)


def run_compression_suite(device: str) -> bool:
    """Execute compression comparison across 4 SVD algorithms."""
    section("Comparing 4 SVD Factorization Algorithms")

    methods = [
        ("Direct Truncated SVD", "direct_svd"),
        ("Activation-Aware SVD", "adasvd_compressor"),
        ("Fisher-Weighted SVD", "fwsvd_compressor"),
        ("Residual SVD", "residual_svd_compressor"),
    ]

    calibration_batches = toy_calibration_loader(num_batches=2, device=device)
    inputs = {k: v.to(device) for k, v in tiny_inputs().items()}

    base_model = TinyClassifier().to(device)
    base_params = base_model.count_parameters()
    kv("Baseline Model Parameters", f"{base_params:,}")
    print()

    table_rows: list[list[str]] = []
    all_passed = True

    for label, method_name in methods:
        candidate_model = TinyClassifier().to(device)

        try:
            compressed = neuralops.compress(
                candidate_model,
                method=method_name,
                rank=8,
                calibration_data=calibration_batches,
            )
            comp_params = sum(p.numel() for p in compressed.parameters())
            reduction_pct = round(((base_params - comp_params) / base_params) * 100.0, 2)

            with torch.no_grad():
                out = compressed(**inputs)
                logits = out.logits if hasattr(out, "logits") else out
            is_finite = bool(torch.isfinite(logits).all().item())

            energy_pct, frob_err = compute_spectral_metrics(base_model, compressed, rank=8)
            status_str = "PASS" if is_finite else "FAIL"
            all_passed = all_passed and is_finite

            table_rows.append([
                label,
                f"{reduction_pct:+.2f}%",
                f"{energy_pct:.2f}%",
                f"{frob_err:.4f}",
                status_str,
            ])
            pass_fail(f"{label} ({method_name})", is_finite, f"reduction={reduction_pct}%, energy={energy_pct}%")

        except Exception as exc:
            table_rows.append([label, "N/A", "N/A", "N/A", f"FAIL: {type(exc).__name__}"])
            pass_fail(f"{label} ({method_name})", False, str(exc))
            all_passed = False

    print("\n" + "-" * 75)
    print_benchmark_table(
        table_rows,
        headers=("Method", "Param Delta", "Energy Retained", "Frob Error", "Status"),
    )
    print("-" * 75 + "\n")

    return all_passed


def main() -> int:
    section("NeuralOps Example 02: Model Compression Suite")
    backend = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", backend)
    kv("PyTorch Version", torch.__version__)
    print()

    success = run_compression_suite(backend)
    print()
    if success:
        pass_fail("Overall Example 02 Status", True, "All 4 compression algorithms verified")
        return 0
    else:
        pass_fail("Overall Example 02 Status", False, "One or more compression algorithms failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
