// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void admm_2of4_fwd(
    const float* __restrict__ W,
    float* __restrict__ out,
    int total_groups
) {
    int pid = blockIdx.x * blockDim.x + threadIdx.x;
    if (pid >= total_groups) return;

    int base = pid * 4;
    float w0 = W[base + 0];
    float w1 = W[base + 1];
    float w2 = W[base + 2];
    float w3 = W[base + 3];

    float s0 = fabsf(w0);
    float s1 = fabsf(w1);
    float s2 = fabsf(w2);
    float s3 = fabsf(w3);

    int r0 = (s1 > s0) + (s2 > s0) + (s3 > s0);
    int r1 = (s0 >= s1) + (s2 > s1) + (s3 > s1);
    int r2 = (s0 >= s2) + (s1 >= s2) + (s3 > s2);
    int r3 = (s0 >= s3) + (s1 >= s3) + (s2 >= s3);

    out[base + 0] = (r0 < 2) ? w0 : 0.0f;
    out[base + 1] = (r1 < 2) ? w1 : 0.0f;
    out[base + 2] = (r2 < 2) ? w2 : 0.0f;
    out[base + 3] = (r3 < 2) ? w3 : 0.0f;
}

torch::Tensor admm_2of4_forward(
    const torch::Tensor& W
) {
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");
    TORCH_CHECK(W.numel() % 4 == 0, "W.numel() must be divisible by 4");

    int total_elements = W.numel();
    int total_groups = total_elements / 4;
    auto out = torch::empty_like(W);

    int threads = 256;
    int blocks = (total_groups + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    admm_2of4_fwd<<<blocks, threads, 0, stream>>>(
        W.data_ptr<float>(),
        out.data_ptr<float>(),
        total_groups
    );

    return out;
}
