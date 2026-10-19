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
__global__ void __launch_bounds__(128) jamba_info_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ w,
    const int B,
    const int N,
    const int D,
    const int K,
    const float eps
) {
    const int b = blockIdx.y;
    if (b >= B) return;

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= N) return;

    const int x_offset = b * N * D + t * D;
    const int out_offset = b * N * K + t * K;

    for (int k = 0; k < K; ++k) {
        float acc = 0.0f;
        const int w_offset = k * D;
        for (int d = 0; d < D; ++d) {
            const float x_val = to_float(x[x_offset + d]);
            const float w_val = to_float(w[w_offset + d]);
            acc += x_val * w_val;
        }

        const float c = 0.5f * acc * (1.0f + tanhf(0.7978845608f * (acc + 0.044715f * acc * acc * acc)));
        out[out_offset + k] = from_float<scalar_t>(c);
    }
}

}}

torch::Tensor jamba_info_fwd(
    const torch::Tensor& x,
    const torch::Tensor& bottleneck_weights,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(bottleneck_weights);

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    const int K = bottleneck_weights.size(0);

    auto out = torch::empty({B, N, K}, x.options());
    if (B == 0 || N == 0 || K == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (N + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto w_c = bottleneck_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "jamba_info_fwd", ([&] {
            neuralops::cuda::jamba_info_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                B, N, D, K, static_cast<float>(eps)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
