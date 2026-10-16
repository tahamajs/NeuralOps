"""
Phase 1 Smoke & Invariant Test Suite for CUDA C++ Infrastructure.
Verifies compilation, caching, SM detection, and the 3-level fallback chain.
"""

from unittest import mock
import pytest
import torch

from neuralops.core.cuda_kernels._bindings import load_cuda_module, get_cuda_arch_flags
from neuralops.core.cuda_kernels._fallback import dispatch


def test_cuda_arch_flags_detection():
    flags = get_cuda_arch_flags()
    assert isinstance(flags, list)
    assert len(flags) > 0
    assert flags[0].startswith("-arch=sm_")


def test_cuda_smoke_double_it():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available for smoke test")

    cpp_src = "torch::Tensor double_it(torch::Tensor x);"
    cuda_src = r"""
    __global__ void double_it_kernel(float* x, int n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            x[idx] *= 2.0f;
        }
    }
    torch::Tensor double_it(torch::Tensor x) {
        int n = x.numel();
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        double_it_kernel<<<blocks, threads>>>(x.data_ptr<float>(), n);
        return x;
    }
    """
    mod = load_cuda_module(
        name="test_infra_double_it",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["double_it"],
    )
    assert mod is not None, "Failed to load CUDA C++ smoke module"

    x = torch.ones(512, device="cuda", dtype=torch.float32)
    out = mod.double_it(x)
    torch.cuda.synchronize()
    assert out[0].item() == 2.0, f"Expected 2.0, got {out[0].item()}"


def test_fallback_chain_all_levels():
    def pytorch_ref(x):
        return x + 1.0

    def triton_mock(x):
        return x + 2.0

    def cuda_mock(x):
        return x + 3.0

    def cuda_failing(x):
        raise RuntimeError("CUDA out of memory simulation")

    def triton_failing(x):
        raise RuntimeError("Triton JIT compilation error")

    x = torch.zeros(4)

    # 1. Normal CUDA execution
    res1 = dispatch(cuda_mock, triton_mock, pytorch_ref, x)
    assert torch.allclose(res1, x + 3.0)

    # 2. CUDA fails -> Triton succeeds
    res2 = dispatch(cuda_failing, triton_mock, pytorch_ref, x)
    assert torch.allclose(res2, x + 2.0)

    # 3. CUDA & Triton fail -> PyTorch succeeds
    res3 = dispatch(cuda_failing, triton_failing, pytorch_ref, x)
    assert torch.allclose(res3, x + 1.0)


def test_fallback_when_cuda_unavailable():
    with mock.patch("torch.cuda.is_available", return_value=False):
        mod = load_cuda_module(
            name="test_no_cuda",
            cpp_sources="",
            cuda_sources="",
            functions=[],
        )
        assert mod is None


def test_fallback_when_pascal_sm6():
    with mock.patch("torch.cuda.get_device_capability", return_value=(6, 1)):
        mod = load_cuda_module(
            name="test_pascal_sm6",
            cpp_sources="",
            cuda_sources="",
            functions=[],
        )
        assert mod is None


def test_fallback_when_no_nvcc():
    with mock.patch("shutil.which", return_value=None):
        mod = load_cuda_module(
            name="test_no_nvcc",
            cpp_sources="",
            cuda_sources="",
            functions=[],
        )
        assert mod is None
