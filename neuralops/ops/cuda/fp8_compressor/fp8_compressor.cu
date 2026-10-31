// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

__global__ void fp8_compress_fwd(
    const float* __restrict__ W,
    float* __restrict__ out,
    int N,
    float scale
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float max_val = 448.0f;
    const float mantissa_step = 0.125f;

    float w = W[idx];
    float w_scaled = fminf(fmaxf(w / scale, -max_val), max_val);
    float abs_w = fmaxf(fabsf(w_scaled), 1e-8f);
    float exp_val = floorf(log2f(abs_w));
    float base = exp2f(exp_val);
    float step = base * mantissa_step;
    float w_fp8 = rintf(w_scaled / step) * step;
    out[idx] = w_fp8 * scale;
}

torch::Tensor fp8_compress_forward(
    const torch::Tensor& W,
    double scale
) {
    TORCH_CHECK(W.is_cuda(), "W must be a CUDA tensor");
    TORCH_CHECK(W.is_contiguous(), "W must be contiguous");
    TORCH_CHECK(W.scalar_type() == torch::kFloat32, "W must be float32");

    int N = W.numel();
    auto out = torch::empty_like(W);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    fp8_compress_fwd<<<blocks, threads, 0, stream>>>(
        W.data_ptr<float>(),
        out.data_ptr<float>(),
        N,
        static_cast<float>(scale)
    );

    return out;
}
