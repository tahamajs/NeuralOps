#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Tests for neuralops.core.roofline real metric calculations."""

from __future__ import annotations

import pytest

from neuralops.core.roofline import (
    RooflineReport,
    estimate_module_flops_and_bytes,
    get_device_roofline_specs,
    profile_roofline,
    render_roofline_terminal,
)


def test_get_device_roofline_specs() -> None:
    """Verify get_device_roofline_specs returns non-zero real specs."""
    dev_name, sm_ver, peak_tflops, peak_bw = get_device_roofline_specs("cpu")
    assert "cpu" in sm_ver.lower()
    assert peak_tflops > 0.0
    assert peak_bw > 0.0


def test_estimate_module_flops_and_bytes_dynamic() -> None:
    """Verify FLOPs and memory traffic scale quadratically/linearly with sequence length."""
    from neuralops.api import create_attention

    module = create_attention("full_attention", embed_dim=128, num_heads=4)
    f1, b_naive1, b_fused1 = estimate_module_flops_and_bytes(
        module=module,
        batch_size=2,
        seq_len=64,
        embed_dim=128,
        num_heads=4,
        element_bytes=2,
    )
    f2, b_naive2, b_fused2 = estimate_module_flops_and_bytes(
        module=module,
        batch_size=2,
        seq_len=128,
        embed_dim=128,
        num_heads=4,
        element_bytes=2,
    )
    # FLOPs grow when sequence length doubles
    assert f2 > 2 * f1
    assert b_naive2 > b_naive1
    assert b_fused2 > 0.0


def test_profile_roofline_computes_real_metrics() -> None:
    """Verify profile_roofline returns a valid RooflineReport with real measured values."""
    report = profile_roofline(
        "full_attention",
        batch_size=1,
        seq_len=16,
        embed_dim=32,
        num_heads=2,
        warmup=2,
        rep=2,
        device="cpu",
        log_to_ledger=False,
    )
    assert isinstance(report, RooflineReport)
    assert report.module_name == "full_attention"
    assert report.total_flops > 0.0
    assert report.latency_triton_ms > 0.0
    assert report.arithmetic_intensity_naive > 0.0
    assert report.bottleneck_regime != ""
    assert isinstance(report.to_dict(), dict)


def test_render_roofline_terminal_produces_ascii_chart() -> None:
    """Verify terminal rendering produces valid log-log characteristic chart."""
    report = profile_roofline(
        "full_attention",
        batch_size=1,
        seq_len=16,
        embed_dim=32,
        num_heads=2,
        warmup=1,
        rep=1,
        device="cpu",
        log_to_ledger=False,
    )
    chart = render_roofline_terminal(report)
    assert "ROOFLINE & ARITHMETIC INTENSITY REPORT" in chart
    assert "LOG-LOG ROOFLINE HARDWARE CHARACTERISTIC CURVE" in chart
    assert "TFLOPS" in chart
