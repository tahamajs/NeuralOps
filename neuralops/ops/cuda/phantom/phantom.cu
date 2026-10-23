// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include "_common.cuh"

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) phantom_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ gate,
    const int total_elements,
    const float threshold
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float g_val = 1.0f / (1.0f + expf(-to_float(gate[idx])));
        const float mask = (g_val >= threshold) ? g_val : 0.0f;
        out[idx] = from_float<scalar_t>(x_val * mask);
    }
}

}}

torch::Tensor phantom_fwd(
    const torch::Tensor& x,
    const torch::Tensor& gate,
    double threshold
) {
    CHECK_INPUT(x);
    CHECK_INPUT(gate);

    auto out = torch::empty_like(x);
    const int total = x.numel();
    if (total == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto g_c = gate.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "phantom_fwd", ([&] {
            neuralops::cuda::phantom_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                total,
                static_cast<float>(threshold)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
