#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""Regression tests for strict kernel backend forcing."""

from __future__ import annotations

import pytest
import torch

from neuralops.core.cuda_kernels._fallback import dispatch, force_kernel_backend


def test_forced_cuda_does_not_silently_use_cpu_reference() -> None:
    """A forced accelerator backend must not be bypassed by CPU fast-path routing."""
    calls: list[str] = []

    def cuda_impl(x: torch.Tensor) -> torch.Tensor:
        calls.append("cuda")
        return x + 1

    def triton_impl(x: torch.Tensor) -> torch.Tensor:
        calls.append("triton")
        return x + 2

    def reference(x: torch.Tensor) -> torch.Tensor:
        calls.append("pytorch")
        return x + 3

    with force_kernel_backend("cuda"):
        result = dispatch(cuda_impl, triton_impl, reference, torch.tensor([1.0]), kernel_name="forced_cpu")

    assert torch.equal(result, torch.tensor([2.0]))
    assert calls == ["cuda"]


@pytest.mark.parametrize(
    ("backend", "cuda_impl", "triton_impl"),
    [
        ("cuda", None, lambda x: x + 2),
        ("triton", lambda x: x + 1, None),
    ],
)
def test_forced_backend_rejects_missing_implementation(
    backend: str,
    cuda_impl,
    triton_impl,
) -> None:
    """A missing forced implementation must raise rather than falling through."""
    with force_kernel_backend(backend):
        with pytest.raises(RuntimeError, match=f"forced to '{backend}'"):
            dispatch(cuda_impl, triton_impl, lambda x: x + 3, torch.tensor([1.0]), kernel_name="missing")


@pytest.mark.parametrize("backend", ["cuda", "triton"])
def test_forced_backend_rejects_none_result(backend: str) -> None:
    """A forced implementation returning None must not silently invoke another tier."""
    calls: list[str] = []

    def selected(_x: torch.Tensor):
        calls.append(backend)
        return None

    def other(_x: torch.Tensor):
        calls.append("other")
        return torch.tensor([2.0])

    cuda_impl = selected if backend == "cuda" else other
    triton_impl = selected if backend == "triton" else other
    with force_kernel_backend(backend):
        with pytest.raises(RuntimeError, match="returned None"):
            dispatch(cuda_impl, triton_impl, lambda x: x + 3, torch.tensor([1.0]), kernel_name="none_result")

    assert calls == [backend]


def test_force_kernel_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError, match="Unknown backend"):
        with force_kernel_backend("unknown"):
            pass
