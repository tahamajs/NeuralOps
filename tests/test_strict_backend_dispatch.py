"""Contract tests for explicit and automatic backend selection."""

import pytest
import torch

from neuralops.core.cuda_kernels._fallback import (
    dispatch,
    force_kernel_backend,
    get_dispatch_stats,
    reset_dispatch_stats,
)


def _cuda_impl(x: torch.Tensor) -> torch.Tensor:
    return x + 1


def _triton_impl(x: torch.Tensor) -> torch.Tensor:
    return x + 2


def _reference(x: torch.Tensor) -> torch.Tensor:
    return x + 3


def test_cuda_forcing_selects_cuda_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_dispatch_stats()
    x = torch.tensor([1.0])
    monkeypatch.setattr(
        "neuralops.core.cuda_kernels._fallback._is_first_tensor_cpu",
        lambda *_args, **_kwargs: False,
    )
    with force_kernel_backend("cuda"):
        result = dispatch(_cuda_impl, _triton_impl, _reference, x, kernel_name="test")
    assert torch.equal(result, torch.tensor([2.0]))
    assert get_dispatch_stats()["cuda"] == 1


def test_pytorch_forcing_selects_reference_backend() -> None:
    reset_dispatch_stats()
    x = torch.tensor([1.0])
    with force_kernel_backend("pytorch"):
        result = dispatch(_cuda_impl, _triton_impl, _reference, x, kernel_name="test")
    assert torch.equal(result, torch.tensor([4.0]))
    assert get_dispatch_stats()["pytorch"] == 1


def test_auto_fallback_after_cuda_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    reset_dispatch_stats()
    monkeypatch.setattr(
        "neuralops.core.cuda_kernels._fallback._is_first_tensor_cpu",
        lambda *_args, **_kwargs: False,
    )

    def failing_cuda(_: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("simulated CUDA failure")

    x = torch.tensor([1.0])
    with force_kernel_backend("auto"):
        result = dispatch(failing_cuda, None, _reference, x, kernel_name="test")
    assert torch.equal(result, torch.tensor([4.0]))
    stats = get_dispatch_stats()
    assert stats["cuda_failures"] == 1
    assert stats["pytorch"] == 1
