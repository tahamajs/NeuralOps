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
__global__ void __launch_bounds__(256) fast_gelu_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float x2 = x_val * x_val;
        const float poly = fmaf(0.044715f, x2, 1.0f) * x_val;
        const float inner = 0.7978845608f * poly;
        const float y_val = 0.5f * x_val * (1.0f + tanhf(inner));
        out[idx] = from_float<scalar_t>(y_val);
    }
}

}}

torch::Tensor fast_gelu_fwd(const torch::Tensor& x) {
    CHECK_INPUT(x);

    auto out = torch::empty_like(x);
    const int total_elements = x.numel();
    if (total_elements == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total_elements + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "fast_gelu_fwd", ([&] {
            neuralops::cuda::fast_gelu_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                total_elements
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
