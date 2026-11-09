import os
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = """
#include <cuda_runtime.h>
__global__ void add_kernel(float* out, float val) {
    out[threadIdx.x] = threadIdx.x + val;
}
torch::Tensor test_cuda(float val) {
    auto out = torch::zeros({8}, torch::dtype(torch::kFloat32).device(torch::kCUDA));
    add_kernel<<<1, 8>>>(out.data_ptr<float>(), val);
    return out;
}
"""

cpp_src = """
#include <torch/extension.h>
torch::Tensor test_cuda(float val);
"""

print("Starting CUDA extension compilation test...")
try:
    mod = load_inline(
        name="test_modern_ld",
        cpp_sources=cpp_src,
        cuda_sources=cuda_src,
        functions=["test_cuda"],
        verbose=True,
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"]
    )
    result = mod.test_cuda(42.0)
    print("SUCCESS! Output tensor:", result)
except Exception as e:
    print("FAILED with error:", e)
