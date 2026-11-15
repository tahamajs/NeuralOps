#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Tests for Example 11: Quantization Formats Zoo (FP8, INT4, MXFP8).
"""

from __future__ import annotations

import importlib
import subprocess
import sys
import torch

from examples._shared import TinyClassifier

ex11 = importlib.import_module("examples.11_quantization_fp8_int4_mx")
compute_sqnr = ex11.compute_sqnr
quantize_fp8_e4m3 = ex11.quantize_fp8_e4m3
quantize_int4_groupwise = ex11.quantize_int4_groupwise
quantize_microscaling_mx = ex11.quantize_microscaling_mx


def test_11_runs_clean():
    """Verify that example 11 executes cleanly as a subprocess and exits 0."""
    proc = subprocess.run(
        [sys.executable, "examples/11_quantization_fp8_int4_mx.py"],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert proc.returncode == 0, f"Example 11 failed with stderr: {proc.stderr}"
    assert "[PASS] Overall Example 11 Status" in proc.stdout


def test_11_sqnr_threshold_enforced():
    """Verify SQNR calculations meet standard numerical precision thresholds."""
    model = TinyClassifier().to("cpu")
    weight = model.transformer.distilbert.transformer.layer[0].ffn.lin1.weight.detach().float()

    # 1. FP8 E4M3 quantization SQNR threshold (>= 20 dB)
    w_fp8, sqnr_fp8, err_fp8, fp8_ok = quantize_fp8_e4m3(weight)
    assert fp8_ok, "FP8 quantization simulation failed"
    assert sqnr_fp8 >= 20.0, f"FP8 SQNR was {sqnr_fp8:.2f} dB, expected >= 20.0 dB"
    assert torch.isfinite(w_fp8).all(), "FP8 reconstructed weight contains non-finite values"

    # 2. INT4 group-wise quantization SQNR threshold (>= 15 dB)
    w_int4, sqnr_int4, err_int4 = quantize_int4_groupwise(weight, group_size=32)
    assert sqnr_int4 >= 15.0, f"INT4 SQNR was {sqnr_int4:.2f} dB, expected >= 15.0 dB"
    assert torch.isfinite(w_int4).all(), "INT4 reconstructed weight contains non-finite values"

    # 3. Microscaling MXFP8 quantization SQNR threshold (>= 20 dB)
    w_mx, sqnr_mx, err_mx = quantize_microscaling_mx(weight, block_size=32)
    assert sqnr_mx >= 20.0, f"MXFP8 SQNR was {sqnr_mx:.2f} dB, expected >= 20.0 dB"
    assert torch.isfinite(w_mx).all(), "MX reconstructed weight contains non-finite values"
