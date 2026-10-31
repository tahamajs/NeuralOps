// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void awq_compress_fwd(
    const float* __restrict__ W,
    const float* __restrict__ scales,
    float* __restrict__ out,
    int N,
    int group_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int group_idx = (group_size > 0) ? (idx / group_size) : idx;
    float s = scales[group_idx];
    float w = W[idx];
    float q = rintf(w / s);
    q = fminf(fmaxf(q, -8.0f), 7.0f);
    out[idx] = q * s;
}

torch::Tensor awq_compress_forward(
    const torch::Tensor& W,
    const torch::Tensor& scales,
    int64_t group_size
) {
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(scales.is_cuda(), "scales must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(scales.is_contiguous(), "scales must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32, "scales must be float32");

    int N = W.numel();
    auto out = torch::empty_like(W);

    int actual_group_size = (scales.numel() == N) ? 1 : static_cast<int>(group_size);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    awq_compress_fwd<<<blocks, threads, 0, stream>>>(
        W.data_ptr<float>(),
        scales.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        actual_group_size
    );

    return out;
}
