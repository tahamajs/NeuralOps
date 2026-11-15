#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Unit and contract tests for examples/_shared foundation utilities.
"""

from __future__ import annotations

import io
from contextlib import redirect_stdout
import pytest
import torch

from examples._shared import (
    TinyClassifier,
    assert_parity,
    assert_shape_unchanged,
    get_backend_preference,
    gpu_available,
    gpu_name,
    kv,
    measure_latency,
    pass_fail,
    section,
    skip,
    skip_if_missing,
    table,
    tiny_inputs,
    toy_dataloader,
)


def test_tiny_classifier_is_real_module():
    m = TinyClassifier()
    assert isinstance(m, torch.nn.Module)
    total_params = m.count_parameters()
    trainable_params = m.count_trainable_parameters()
    assert total_params > 0
    assert trainable_params == total_params

    # Forward pass
    inputs = tiny_inputs(batch_size=2, seq_len=8)
    out = m(**inputs)
    assert out is not None
    assert hasattr(out, "logits")
    assert out.logits.shape[0] == 2


def test_pass_fail_requires_bool():
    with pytest.raises(TypeError):
        pass_fail("test", "truthy_string", "detail")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pass_fail("test", 1, "detail")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        pass_fail("test", None, "detail")  # type: ignore[arg-type]


def test_pass_fail_output_capture():
    buf = io.StringIO()
    with redirect_stdout(buf):
        pass_fail("test_check", True, "detail_ok")
    assert "[PASS]" in buf.getvalue()
    assert "detail_ok" in buf.getvalue()

    buf = io.StringIO()
    with redirect_stdout(buf):
        pass_fail("test_check", False, "detail_fail")
    assert "[FAIL]" in buf.getvalue()
    assert "detail_fail" in buf.getvalue()


def test_assert_parity_behavior():
    a = torch.zeros(4, 4)
    b = torch.ones(4, 4)
    # Different tensors should return False (and not raise unhandled exception)
    res = assert_parity(a, b, atol=1e-3, label="mismatch_test")
    assert res is False

    c = a.clone()
    res_ok = assert_parity(a, c, atol=1e-3, label="match_test")
    assert res_ok is True


def test_measure_latency_positive():
    def compute():
        return torch.matmul(torch.ones(32, 32), torch.ones(32, 32))

    lat = measure_latency(compute, warmup=2, iters=5)
    assert lat > 0.0


def test_backend_preference_valid():
    backend = get_backend_preference()
    assert backend in ("cuda", "mps", "cpu")


def test_toy_dataloader_yields_batches():
    loader = toy_dataloader(num_batches=2, batch_size=4)
    batches = list(loader)
    assert len(batches) == 2
    for b in batches:
        assert "input_ids" in b
        assert "attention_mask" in b
        assert "labels" in b
        assert b["input_ids"].shape == (4, 16)
        assert b["labels"].shape == (4,)


def test_reporting_helpers():
    buf = io.StringIO()
    with redirect_stdout(buf):
        section("Test Section")
        kv("Key", "Value")
        table([["row1_col1", "row1_col2"]], headers=["H1", "H2"])
        skip("Optional", "Package missing")
    output = buf.getvalue()
    assert "Test Section" in output
    assert "Key" in output
    assert "H1" in output
    assert "[SKIPPED]" in output
