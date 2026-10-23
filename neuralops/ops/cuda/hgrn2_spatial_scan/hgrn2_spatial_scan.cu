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
__global__ void __launch_bounds__(256) hgrn2_spatial_scan_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    const int b = bh / H;
    const int h = bh % H;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || d >= D) return;

    float s = 0.0f;
    for (int t = 0; t < N; ++t) {
        const int idx = (b * H + h) * N * D + t * D + d;
        const float gate = 1.0f / (1.0f + expf(-to_float(g[idx])));
        const float x_val = to_float(x[idx]);

        s = gate * s + (1.0f - gate) * x_val;
        out[idx] = from_float<scalar_t>(s);
    }
}

}}

torch::Tensor hgrn2_spatial_scan_fwd(
    const torch::Tensor& x,
    const torch::Tensor& g
) {
    CHECK_INPUT(x);
    CHECK_INPUT(g);

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    auto out = torch::empty_like(x);

    if (B == 0 || H == 0 || D == 0) return out;

    constexpr int THREADS = 256;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto g_c = g.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hgrn2_spatial_scan_fwd", ([&] {
            neuralops::cuda::hgrn2_spatial_scan_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
