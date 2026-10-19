// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/util/Optional.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__launch_bounds__(256)
__global__ void causal_conv1d_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    const scalar_t* __restrict__ bias,
    const int B,
    const int D,
    const int L,
    const int Kw,
    const bool has_bias,
    const bool silu_activation
) {
    const int total = B * D * L;

    for (int idx = blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += blockDim.x * gridDim.x) {
        const int b = idx / (D * L);
        const int rem = idx % (D * L);
        const int d = rem / L;
        const int t = rem % L;

        float acc = (has_bias && bias != nullptr) ? to_float(bias[d]) : 0.0f;
        const int row_base = b * (D * L) + d * L;
        const int weight_base = d * Kw;

        #pragma unroll 4
        for (int k = 0; k < Kw; ++k) {
            const int in_t = t - (Kw - 1 - k);
            if (in_t >= 0 && in_t < L) {
                const float x_val = to_float(x[row_base + in_t]);
                const float w_val = to_float(weight[weight_base + k]);
                acc = fmaf(x_val, w_val, acc);
            }
        }

        if (silu_activation) {
            const float sig = 1.0f / (1.0f + expf(-acc));
            acc = acc * sig;
        }

        out[idx] = from_float<scalar_t>(acc);
    }
}

}
}

torch::Tensor causal_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const bool silu_activation
) {
    TORCH_CHECK(x.is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(weight.is_cuda(), "weight must be a CUDA tensor");

    const int B = x.size(0);
    const int D = x.size(1);
    const int L = x.size(2);
    const int Kw = weight.size(1);

    auto out = torch::empty_like(x);
    const int total = B * D * L;
    if (total == 0) return out;

    const bool has_bias = bias.has_value() && bias.value().defined();
    constexpr int THREADS = 256;
    const int blocks = (total + THREADS - 1) / THREADS;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x.contiguous();
    auto w_c = weight.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "causal_conv1d_fwd", ([&] {
            neuralops::cuda::causal_conv1d_fwd_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                has_bias ? bias.value().contiguous().data_ptr<scalar_t>() : nullptr,
                B, D, L, Kw, has_bias, silu_activation
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
