#!/usr/bin/env python3
# Copyright (c) 2024-2026, Taha Majlesi
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# For a list of all contributors, visit:
#   https://github.com/tahamajs/NeuralOps/graphs/contributors
"""
Phase 11: 3-Level Fallback Dispatch and Hardware Forcing Verification Suite.
Validates:
1. `force_kernel_backend("pytorch")` directs execution strictly to PyTorch reference.
2. `force_kernel_backend("triton")` executes Triton kernel when available.
3. `force_kernel_backend("cuda")` executes CUDA or fails closed with descriptive RuntimeError.
4. Forcing 'cuda' when CUDA is unavailable fails immediately with NO silent fallback to PyTorch.
5. Telemetry counter verification via `get_dispatch_stats()`.
"""

from unittest.mock import patch
import pytest
import torch
from neuralops.ops.cuda._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
)


def mock_cuda_kernel(x):
    return x * 2.0


def mock_triton_kernel(x):
    return x * 3.0


def mock_pytorch_kernel(x):
    return x * 4.0


@pytest.fixture(autouse=True)
def clean_dispatch_state():
    reset_dispatch_stats()
    yield
    reset_dispatch_stats()


def test_force_pytorch_backend():
    """Verify force_kernel_backend('pytorch') routes to PyTorch reference."""
    reset_dispatch_stats()
    x = torch.randn(4, 4)
    with force_kernel_backend("pytorch"):
        out = dispatch(
            mock_cuda_kernel,
            mock_triton_kernel,
            mock_pytorch_kernel,
            x,
            kernel_name="test_kernel",
        )
    assert torch.allclose(out, x * 4.0)
    stats = get_dispatch_stats()
    assert stats["pytorch"] == 1
    assert stats["cuda"] == 0
    assert stats["triton"] == 0


def test_force_triton_backend_success():
    """Verify force_kernel_backend('triton') routes to Triton kernel."""
    reset_dispatch_stats()
    x = torch.randn(4, 4)
    with force_kernel_backend("triton"):
        out = dispatch(
            mock_cuda_kernel,
            mock_triton_kernel,
            mock_pytorch_kernel,
            x,
            kernel_name="test_kernel",
        )
    assert torch.allclose(out, x * 3.0)
    stats = get_dispatch_stats()
    assert stats["triton"] == 1
    assert stats["pytorch"] == 0


def test_force_triton_backend_fail_closed():
    """Verify force_kernel_backend('triton') fails closed when triton_fn is None."""
    reset_dispatch_stats()
    x = torch.randn(4, 4)
    with force_kernel_backend("triton"):
        with pytest.raises(RuntimeError) as exc_info:
            dispatch(
                mock_cuda_kernel,
                None,
                mock_pytorch_kernel,
                x,
                kernel_name="test_triton_missing",
            )
    assert "Backend forced to 'triton', but no Triton implementation is available" in str(exc_info.value)
    stats = get_dispatch_stats()
    assert stats["triton_failures"] == 1
    assert stats["pytorch"] == 0


def test_force_cuda_backend_success():
    """Verify force_kernel_backend('cuda') routes to CUDA C++ kernel."""
    reset_dispatch_stats()
    x = torch.randn(4, 4)
    with force_kernel_backend("cuda"):
        out = dispatch(
            mock_cuda_kernel,
            mock_triton_kernel,
            mock_pytorch_kernel,
            x,
            kernel_name="test_kernel",
        )
    assert torch.allclose(out, x * 2.0)
    stats = get_dispatch_stats()
    assert stats["cuda"] == 1
    assert stats["pytorch"] == 0


def test_force_cuda_backend_missing_fail_closed():
    """Verify force_kernel_backend('cuda') fails closed when cuda_fn is None."""
    reset_dispatch_stats()
    x = torch.randn(4, 4)
    with force_kernel_backend("cuda"):
        with pytest.raises(RuntimeError) as exc_info:
            dispatch(
                None,
                mock_triton_kernel,
                mock_pytorch_kernel,
                x,
                kernel_name="test_cuda_missing",
            )
    assert "Backend forced to 'cuda', but no CUDA C++ implementation is available" in str(exc_info.value)
    stats = get_dispatch_stats()
    assert stats["cuda_failures"] == 1
    assert stats["pytorch"] == 0  # CRITICAL: zero silent fallback to PyTorch allowed!


def test_force_cuda_execution_failure_fail_closed():
    """Verify that when a CUDA kernel crashes, forced backend raises RuntimeError rather than falling back to CPU."""
    reset_dispatch_stats()
    def broken_cuda_kernel(x):
        raise RuntimeError("CUDA out of memory or device side assert")

    x = torch.randn(4, 4)
    with force_kernel_backend("cuda"):
        with pytest.raises(RuntimeError) as exc_info:
            dispatch(
                broken_cuda_kernel,
                mock_triton_kernel,
                mock_pytorch_kernel,
                x,
                kernel_name="test_broken_cuda",
            )
    assert "Backend forced to 'cuda', but execution failed" in str(exc_info.value)
    stats = get_dispatch_stats()
    assert stats["cuda_failures"] == 1
    assert stats["pytorch"] == 0


def test_force_cuda_simulated_unavailable():
    """Simulate torch.cuda.is_available()=False and verify that forcing cuda does not silently use PyTorch."""
    reset_dispatch_stats()
    def cuda_only_kernel(x):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available on this platform.")
        return x * 2.0

    x = torch.randn(4, 4)
    with patch("torch.cuda.is_available", return_value=False):
        with force_kernel_backend("cuda"):
            with pytest.raises(RuntimeError) as exc_info:
                dispatch(
                    cuda_only_kernel,
                    mock_triton_kernel,
                    mock_pytorch_kernel,
                    x,
                    kernel_name="test_simulated_cuda_unavailable",
                )
        assert "Backend forced to 'cuda', but execution failed" in str(exc_info.value)
        assert "CUDA is not available on this platform" in str(exc_info.value)

    stats = get_dispatch_stats()
    assert stats["cuda_failures"] == 1
    assert stats["pytorch"] == 0


if __name__ == "__main__":
    reset_dispatch_stats()
    test_force_pytorch_backend()
    print("✓ test_force_pytorch_backend passed")
    test_force_triton_backend_success()
    print("✓ test_force_triton_backend_success passed")
    test_force_triton_backend_fail_closed()
    print("✓ test_force_triton_backend_fail_closed passed")
    test_force_cuda_backend_success()
    print("✓ test_force_cuda_backend_success passed")
    test_force_cuda_backend_missing_fail_closed()
    print("✓ test_force_cuda_backend_missing_fail_closed passed")
    test_force_cuda_execution_failure_fail_closed()
    print("✓ test_force_cuda_execution_failure_fail_closed passed")
    test_force_cuda_simulated_unavailable()
    print("✓ test_force_cuda_simulated_unavailable passed")
    print("\nALL 7 DISPATCH FORCING AND FAIL-CLOSED TESTS PASSED PERFECTLY!")
