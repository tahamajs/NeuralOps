#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Example 18: Three-Level Kernel Fallback Dispatch Hierarchy.

Subsystem Demonstrated:
    The NeuralOps 3-level kernel execution fallback contract:
    - Level 1: Native compiled CUDA C++ / HIP extension (peak hardware throughput)
    - Level 2: Autotuned Triton JIT kernel (vendor-portable accelerator path)
    - Level 3: Vectorized PyTorch reference (universal CPU/MPS fallback)

Execution Environment:
    - CPU: Fully supported, deterministic, offline execution on macOS CPU.
    - GPU: Optional (Level 1 and Live Triton are profiled when CUDA hardware is present).
    - Network: Not required; uses offline synthetic query, key, value tensors.

What You Will Learn:
    1. How the force_kernel_backend() context manager controls routing.
    2. How the dispatcher handles missing CUDA C++ binaries without crashing.
    3. How to inspect global dispatch telemetry using get_dispatch_stats().
    4. How to measure per-backend execution latency and cross-backend numerical parity.
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
    assert_parity,
    get_backend_preference,
    gpu_name,
    kv,
    measure_latency,
    pass_fail,
    section,
    table,
)
from neuralops import force_kernel_backend
from neuralops.ops.cuda import (
    cosformer_linear_attention,
    get_dispatch_stats,
    reset_dispatch_stats,
)


def main() -> int:
    section("NeuralOps Example 18: Three-Level Kernel Dispatch Hierarchy")

    device = get_backend_preference()
    kv("Hardware Accelerator", gpu_name())
    kv("Target Device", device)
    kv("PyTorch Version", torch.__version__)

    # Reset global dispatch counters
    reset_dispatch_stats()

    # Deterministic test tensors: B=2, H=4, L=16, D=32
    torch.manual_seed(42)
    b, h, seq_len, d_head = 2, 4, 16, 32
    # Structured discrete input scaling to avoid unnormalized random evaluation tokens
    q = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32, device=device) / 10.0
    k = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32, device=device) / 10.0
    v = torch.randint(1, 10, (b, h, seq_len, d_head), dtype=torch.float32, device=device) / 10.0

    kv("Operator Under Test", "cosformer_linear_attention")
    kv("Tensor Dimensions", f"B={b}, H={h}, L={seq_len}, D={d_head}")

    dispatch_records: list[dict[str, str | float]] = []
    outputs: dict[str, torch.Tensor] = {}

    p_level1 = False
    p_level2 = False
    p_level3 = False
    cuda_available = torch.cuda.is_available()

    # =========================================================================
    # Level 1: Native CUDA C++ Extension
    # =========================================================================
    section("Level 1: Native Compiled CUDA C++ Extension")
    try:
        with force_kernel_backend("cuda"):
            out_l1 = cosformer_linear_attention(q, k, v)
            outputs["cuda"] = out_l1
            lat_l1 = measure_latency(lambda: cosformer_linear_attention(q, k, v), warmup=2, iters=5)
            p_level1 = bool(torch.isfinite(out_l1).all().item() and out_l1.shape == (b, h, seq_len, d_head))
            pass_fail("Level 1 (CUDA C++) Execution", p_level1, "CUDA C++ extension kernel executed successfully")
            kv("Latency Mean", f"{lat_l1:.4f} ms")
            dispatch_records.append(
                {
                    "Hierarchy Level": "Level 1: CUDA C++",
                    "Backend": "cuda",
                    "Available": "Yes",
                    "Latency (ms)": f"{lat_l1:.4f}",
                    "Status": "PASS",
                }
            )
    except Exception as exc:
        print(f"  [SKIPPED] CUDA not available on this device: {exc}")
        dispatch_records.append(
            {
                "Hierarchy Level": "Level 1: CUDA C++",
                "Backend": "cuda",
                "Available": "No (GPU required)",
                "Latency (ms)": "N/A",
                "Status": "SKIPPED: CUDA binary missing on host",
            }
        )

    # =========================================================================
    # Level 2: Triton JIT Kernel
    # =========================================================================
    section("Level 2: Autotuned Triton JIT Kernel")
    try:
        with force_kernel_backend("triton"):
            out_l2 = cosformer_linear_attention(q, k, v)
            outputs["triton"] = out_l2
            lat_l2 = measure_latency(lambda: cosformer_linear_attention(q, k, v), warmup=2, iters=5)
            p_level2 = bool(torch.isfinite(out_l2).all().item() and out_l2.shape == (b, h, seq_len, d_head))
            pass_fail("Level 2 (Triton JIT) Execution", p_level2, "Triton kernel executed successfully")
            kv("Latency Mean", f"{lat_l2:.4f} ms")
            dispatch_records.append(
                {
                    "Hierarchy Level": "Level 2: Triton JIT",
                    "Backend": "triton",
                    "Available": "Yes",
                    "Latency (ms)": f"{lat_l2:.4f}",
                    "Status": "PASS",
                }
            )
    except Exception as exc:
        print(f"  [SKIPPED] Triton not available on this device: {exc}")
        dispatch_records.append(
            {
                "Hierarchy Level": "Level 2: Triton JIT",
                "Backend": "triton",
                "Available": "No",
                "Latency (ms)": "N/A",
                "Status": f"SKIPPED: {type(exc).__name__}",
            }
        )

    # =========================================================================
    # Level 3: Universal Vectorized PyTorch Reference
    # =========================================================================
    section("Level 3: Universal Vectorized PyTorch Reference")
    with force_kernel_backend("pytorch"):
        out_l3 = cosformer_linear_attention(q, k, v)
        outputs["pytorch"] = out_l3
        lat_l3 = measure_latency(lambda: cosformer_linear_attention(q, k, v), warmup=2, iters=5)
        p_level3 = bool(torch.isfinite(out_l3).all().item() and out_l3.shape == (b, h, seq_len, d_head))
        pass_fail("Level 3 (PyTorch Reference) Execution", p_level3, "PyTorch reference fallback executed successfully")
        kv("Latency Mean", f"{lat_l3:.4f} ms")
        dispatch_records.append(
            {
                "Hierarchy Level": "Level 3: PyTorch Ref",
                "Backend": "pytorch",
                "Available": "Yes (Universal)",
                "Latency (ms)": f"{lat_l3:.4f}",
                "Status": "PASS",
            }
        )

    # =========================================================================
    # Cross-Backend Numerical Parity Verification
    # =========================================================================
    section("Cross-Backend Numerical Parity")
    p_parity = True
    if len(outputs) >= 2:
        keys = list(outputs.keys())
        b1, b2 = keys[0], keys[1]
        p_parity = bool(assert_parity(outputs[b1], outputs[b2], atol=1e-4, label=f"Parity: {b1} vs {b2}"))
    else:
        print(
            f"  [NOTE] Only 1 backend ({list(outputs.keys())}) available on this host; parity check requires >= 2 backends."
        )

    # =========================================================================
    # Dispatch Telemetry & Summary Table
    # =========================================================================
    section("3-Level Dispatch Table Summary")
    headers = ["Hierarchy Level", "Backend", "Available", "Latency (ms)", "Status"]
    rows = [[str(r[h]) for h in headers] for r in dispatch_records]
    table(rows, headers=headers)

    stats = get_dispatch_stats()
    kv("Recorded Dispatch Statistics", stats)

    # Compute overall status from real execution and parity checks
    # Level 3 must pass. If Level 2 passed, parity must hold. If CUDA is present, Level 1 must pass.
    all_passed = bool(
        p_level3
        and (p_level1 or not cuda_available)
        and p_parity
        and (stats.get("pytorch", 0) > 0 or stats.get("triton", 0) > 0)
    )

    pass_fail(
        "Overall Example 18 Status",
        all_passed,
        "3-level fallback hierarchy gracefully probed with verified execution and telemetry"
        if all_passed
        else "Dispatch hierarchy verification failed",
    )
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
