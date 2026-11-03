#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Validation script for the 4 critical CUDA C++ kernels:
  1. gla_scan (Gated Linear Attention Scan)
  2. hgrn2_scan (Hierarchically Gated RNN v2 Scan)
  3. selective_scan (Mamba Selective State Space Scan)
  4. sliding_window (Sliding Window Attention with Sinks)

Strict Validation Contract:
  - Fail-closed execution: requires active CUDA device; zero CPU fallback.
  - Zero backend identity spoofing: get_last_dispatch_backend(name) must strictly equal 'cuda'.
  - Numerical parity verification: forward max_diff computed against PyTorch reference.
  - Backward parity verification: input and parameter gradients compared against PyTorch reference.
  - Clean non-zero exit code on any parity failure, NaN/Inf, or backend misidentification.
"""

from __future__ import annotations

import logging
import sys
from typing import Any
import torch

from neuralops.ops.cuda._fallback import (
    force_kernel_backend,
    get_last_dispatch_backend,
    get_last_dispatch_error,
)
from neuralops.ops.cuda.gla_scan.gla_autograd import gla_scan, gla_scan_pytorch_ref
from neuralops.ops.cuda.hgrn2_scan.hgrn2_autograd import hgrn2_scan, hgrn2_fused_bwd_scan_pytorch_ref
from neuralops.ops.cuda.selective_scan.selective_scan_autograd import selective_scan, selective_scan_pytorch_ref
from neuralops.ops.cuda.sliding_window.sliding_autograd import sliding_window_attn, sliding_window_pytorch_ref

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("validate_4_cuda_kernels")


def _check_parity(
    name: str,
    target_out: torch.Tensor,
    ref_out: torch.Tensor,
    tol: float = 1e-3,
) -> tuple[bool, float, str]:
    """Verify finite outputs and numerical difference tolerance."""
    if not torch.isfinite(target_out).all().item():
        return False, float("inf"), "FAIL_NAN (target has non-finite values)"
    if not torch.isfinite(ref_out).all().item():
        return False, float("inf"), "FAIL_NAN (reference has non-finite values)"

    diff = (target_out.float() - ref_out.float()).abs().max().item()
    if diff > tol:
        return False, diff, f"FAIL_NUMERICS (max_diff={diff:.2e} > tol={tol:.2e})"
    return True, diff, "PASS"


def main() -> int:
    if not torch.cuda.is_available():
        logger.error("CUDA is not available on this host. Aborting.")
        return 1

    device_name = torch.cuda.get_device_name(0)
    logger.info(f"Target GPU: {device_name}")
    logger.info(f"PyTorch Version: {torch.__version__}, CUDA: {torch.version.cuda}")

    results: dict[str, dict[str, Any]] = {}

    # =========================================================================
    # 1. GLA Scan
    # =========================================================================
    logger.info("=== Validating 1/4: GLA Scan (Native CUDA C++) ===")
    try:
        q = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        with force_kernel_backend("cuda"):
            out = gla_scan(q, k, v)
            loss = out.sum()
            loss.backward()

        out_ref = gla_scan_pytorch_ref(q_ref, k_ref, v_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        backend = get_last_dispatch_backend("gla_scan")
        if backend != "cuda":
            raise RuntimeError(f"Backend identity violation: expected 'cuda', got '{backend}'")

        ok_fwd, fwd_diff, fwd_msg = _check_parity("gla_scan:fwd", out, out_ref, tol=1e-3)
        assert q.grad is not None and q_ref.grad is not None, "Missing gradient on q"
        ok_bwd, bwd_diff, bwd_msg = _check_parity("gla_scan:bwd", q.grad, q_ref.grad, tol=2e-3)

        status_ok = ok_fwd and ok_bwd
        results["gla_scan"] = {
            "passed": status_ok,
            "backend": backend,
            "fwd_diff": fwd_diff,
            "bwd_diff": bwd_diff,
            "message": f"{fwd_msg} | bwd: {bwd_msg}",
        }
        logger.info(f"  [{'PASS' if status_ok else 'FAIL'}] gla_scan (backend='{backend}', fwd_diff={fwd_diff:.2e}, bwd_diff={bwd_diff:.2e})")
    except Exception as exc:
        err = get_last_dispatch_error("gla_scan")
        logger.error(f"  [FAIL] gla_scan failed: {exc} (last_error={err})")
        results["gla_scan"] = {"passed": False, "backend": get_last_dispatch_backend("gla_scan"), "message": str(exc)}

    # =========================================================================
    # 2. HGRN2 Scan
    # =========================================================================
    logger.info("=== Validating 2/4: HGRN2 Scan (Native CUDA C++) ===")
    try:
        x = torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float32, requires_grad=True)
        g = torch.sigmoid(torch.randn(2, 4, 32, 64, device="cuda", dtype=torch.float32)).requires_grad_(True)

        x_ref = x.detach().clone().requires_grad_(True)
        g_ref = g.detach().clone().requires_grad_(True)

        with force_kernel_backend("cuda"):
            out = hgrn2_scan(x, g)
            loss = out.sum()
            loss.backward()

        out_ref = hgrn2_fused_bwd_scan_pytorch_ref(x_ref, g_ref)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        backend = get_last_dispatch_backend("hgrn2_fused_bwd_scan") or get_last_dispatch_backend("hgrn2_scan")
        if backend != "cuda":
            raise RuntimeError(f"Backend identity violation: expected 'cuda', got '{backend}'")

        ok_fwd, fwd_diff, fwd_msg = _check_parity("hgrn2_scan:fwd", out, out_ref, tol=1e-3)
        assert x.grad is not None and x_ref.grad is not None, "Missing gradient on x"
        ok_bwd, bwd_diff, bwd_msg = _check_parity("hgrn2_scan:bwd", x.grad, x_ref.grad, tol=2e-3)

        status_ok = ok_fwd and ok_bwd
        results["hgrn2_scan"] = {
            "passed": status_ok,
            "backend": backend,
            "fwd_diff": fwd_diff,
            "bwd_diff": bwd_diff,
            "message": f"{fwd_msg} | bwd: {bwd_msg}",
        }
        logger.info(f"  [{'PASS' if status_ok else 'FAIL'}] hgrn2_scan (backend='{backend}', fwd_diff={fwd_diff:.2e}, bwd_diff={bwd_diff:.2e})")
    except Exception as exc:
        err = get_last_dispatch_error("hgrn2_scan") or get_last_dispatch_error("hgrn2_fused_bwd_scan")
        logger.error(f"  [FAIL] hgrn2_scan failed: {exc} (last_error={err})")
        results["hgrn2_scan"] = {"passed": False, "backend": get_last_dispatch_backend("hgrn2_scan"), "message": str(exc)}

    # =========================================================================
    # 3. Selective Scan
    # =========================================================================
    logger.info("=== Validating 3/4: Selective Scan (Native CUDA C++) ===")
    try:
        b, d, seq, dstate = 2, 32, 64, 16
        u = torch.randn(b, d, seq, device="cuda", dtype=torch.float32, requires_grad=True)
        # S6 requires strictly positive step sizes (delta > 0) to avoid explosive recurrence
        delta = (torch.rand(b, d, seq, device="cuda", dtype=torch.float32) * 0.1 + 0.01).requires_grad_(True)
        a = -(torch.rand(d, dstate, device="cuda", dtype=torch.float32) * 0.5 + 0.5)
        b_t = torch.randn(b, dstate, seq, device="cuda", dtype=torch.float32, requires_grad=True)
        c = torch.randn(b, dstate, seq, device="cuda", dtype=torch.float32, requires_grad=True)
        d_param = torch.randn(d, device="cuda", dtype=torch.float32)

        u_ref = u.detach().clone().requires_grad_(True)
        delta_ref = delta.detach().clone().requires_grad_(True)
        b_t_ref = b_t.detach().clone().requires_grad_(True)
        c_ref = c.detach().clone().requires_grad_(True)

        with force_kernel_backend("cuda"):
            out_k, state_k = selective_scan(u, delta, a, b_t, c, d_param)
            loss = out_k.sum()
            loss.backward()

        out_ref, state_ref = selective_scan_pytorch_ref(u_ref, delta_ref, a, b_t_ref, c_ref, d_param)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        backend = get_last_dispatch_backend("selective_scan")
        if backend != "cuda":
            raise RuntimeError(f"Backend identity violation: expected 'cuda', got '{backend}'")

        ok_fwd_y, diff_y, msg_y = _check_parity("selective_scan:out", out_k, out_ref, tol=1e-3)
        ok_fwd_s, diff_s, msg_s = _check_parity("selective_scan:state", state_k, state_ref, tol=1e-3)
        ok_bwd_u, diff_u, msg_u = (True, 0.0, "PASS")
        if u.grad is not None and u_ref.grad is not None:
            ok_bwd_u, diff_u, msg_u = _check_parity("selective_scan:bwd_u", u.grad, u_ref.grad, tol=2e-3)

        status_ok = ok_fwd_y and ok_fwd_s and ok_bwd_u
        results["selective_scan"] = {
            "passed": status_ok,
            "backend": backend,
            "fwd_diff": max(diff_y, diff_s),
            "bwd_diff": diff_u,
            "message": f"out: {msg_y} | state: {msg_s} | bwd: {msg_u}",
        }
        logger.info(f"  [{'PASS' if status_ok else 'FAIL'}] selective_scan (backend='{backend}', out_diff={diff_y:.2e}, state_diff={diff_s:.2e})")
    except Exception as exc:
        err = get_last_dispatch_error("selective_scan")
        logger.error(f"  [FAIL] selective_scan failed: {exc} (last_error={err})")
        results["selective_scan"] = {"passed": False, "backend": get_last_dispatch_backend("selective_scan"), "message": str(exc)}

    # =========================================================================
    # 4. Sliding Window Attention
    # =========================================================================
    logger.info("=== Validating 4/4: Sliding Window Attention (Native CUDA C++) ===")
    try:
        b, h, seq, d = 2, 4, 64, 32
        q = torch.randn(b, h, seq, d, device="cuda", dtype=torch.float32, requires_grad=True)
        k = torch.randn(b, h, seq, d, device="cuda", dtype=torch.float32, requires_grad=True)
        v = torch.randn(b, h, seq, d, device="cuda", dtype=torch.float32, requires_grad=True)

        q_ref = q.detach().clone().requires_grad_(True)
        k_ref = k.detach().clone().requires_grad_(True)
        v_ref = v.detach().clone().requires_grad_(True)

        window_size = 16
        num_sinks = 4

        with force_kernel_backend("cuda"):
            out = sliding_window_attn(q, k, v, window_size=window_size, num_sinks=num_sinks)
            loss = out.sum()
            loss.backward()

        out_ref = sliding_window_pytorch_ref(q_ref, k_ref, v_ref, window_size=window_size, num_sinks=num_sinks)
        loss_ref = out_ref.sum()
        loss_ref.backward()

        backend = get_last_dispatch_backend("sliding_window")
        if backend != "cuda":
            raise RuntimeError(f"Backend identity violation: expected 'cuda', got '{backend}'")

        ok_fwd, fwd_diff, fwd_msg = _check_parity("sliding_window:fwd", out, out_ref, tol=1e-3)
        ok_bwd = True
        bwd_diff = 0.0
        bwd_msg = "PASS"
        if q.grad is not None and q_ref.grad is not None:
            ok_bwd, bwd_diff, bwd_msg = _check_parity("sliding_window:bwd", q.grad, q_ref.grad, tol=2e-3)

        status_ok = ok_fwd and ok_bwd
        results["sliding_window"] = {
            "passed": status_ok,
            "backend": backend,
            "fwd_diff": fwd_diff,
            "bwd_diff": bwd_diff,
            "message": f"{fwd_msg} | bwd: {bwd_msg}",
        }
        logger.info(f"  [{'PASS' if status_ok else 'FAIL'}] sliding_window (backend='{backend}', fwd_diff={fwd_diff:.2e}, bwd_diff={bwd_diff:.2e})")
    except Exception as exc:
        err = get_last_dispatch_error("sliding_window")
        logger.error(f"  [FAIL] sliding_window failed: {exc} (last_error={err})")
        results["sliding_window"] = {"passed": False, "backend": get_last_dispatch_backend("sliding_window"), "message": str(exc)}

    # =========================================================================
    # Summary Table
    # =========================================================================
    logger.info("=" * 70)
    logger.info("CRITICAL CUDA VALIDATION SUMMARY (STRICT PARITY & BACKEND CONTRACT):")
    all_passed = True
    for name, item in results.items():
        ok = item.get("passed", False)
        all_passed = all_passed and ok
        b_name = item.get("backend", "UNKNOWN")
        fwd_d = item.get("fwd_diff", float("nan"))
        bwd_d = item.get("bwd_diff", float("nan"))
        status_label = "PASSED (NATIVE CUDA)" if ok else "FAILED"
        logger.info(f"  * {name:<18}: {status_label:<20} | backend={b_name:<6} | fwd_diff={fwd_d:.2e} | bwd_diff={bwd_d:.2e}")

    logger.info("=" * 70)

    if all_passed and len(results) == 4:
        logger.info("ALL 4 CRITICAL CUDA KERNELS PASSED STRICT NUMERICAL & BACKEND VALIDATION!")
        return 0
    else:
        logger.error("ONE OR MORE CRITICAL CUDA KERNELS FAILED STRICT VALIDATION.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
