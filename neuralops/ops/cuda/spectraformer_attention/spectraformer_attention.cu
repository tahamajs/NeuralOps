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
__global__ void __launch_bounds__(256) spectraformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ cos_basis,
    const scalar_t* __restrict__ sin_basis,
    const int total_elements
) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < total_elements; idx += stride) {
        const float x_val = to_float(x[idx]);
        const float c_val = to_float(cos_basis[idx]);
        const float s_val = to_float(sin_basis[idx]);

        const float basis_sum = c_val + s_val;
        const float modulated = x_val * basis_sum;

        out[idx] = from_float<scalar_t>(modulated);
    }
}

}}

torch::Tensor spectraformer_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_basis,
    const torch::Tensor& sin_basis
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos_basis);
    CHECK_INPUT(sin_basis);

    TORCH_CHECK(cos_basis.sizes() == x.sizes(), "cos_basis shape mismatch");
    TORCH_CHECK(sin_basis.sizes() == x.sizes(), "sin_basis shape mismatch");

    auto out = torch::empty_like(x);
    const int total = x.numel();
    if (total == 0) return out;

    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (total + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto c_c = cos_basis.contiguous();
    auto s_c = sin_basis.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "spectraformer_fwd", ([&] {
            neuralops::cuda::spectraformer_fwd_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                s_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
