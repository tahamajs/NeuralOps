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
__global__ void __launch_bounds__(128) vspec_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ spec_gate,
    const scalar_t* __restrict__ decay,
    const int B,
    const int N,
    const int D,
    const float eps
) {
    const int b = blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || d >= D) return;

    float s = 0.0f;
    const int base_batch_offset = b * N * D + d;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = base_batch_offset + t * D;
        const float x_val = to_float(x[idx]);
        const float g_raw = to_float(spec_gate[idx]);
        const float g_val = 1.0f / (1.0f + expf(-g_raw));
        const float lam = to_float(decay[idx]);

        s = fmaf(lam, s, x_val * g_val);
        const float norm = s * rsqrtf(fmaf(s, s, eps));
        out[idx] = from_float<scalar_t>(norm);
    }
}

}}

torch::Tensor vspec_fwd(
    const torch::Tensor& x,
    const torch::Tensor& spec_gate,
    const torch::Tensor& decay,
    double eps
) {
    CHECK_INPUT(x);
    CHECK_INPUT(spec_gate);
    CHECK_INPUT(decay);

    TORCH_CHECK(x.dim() == 3, "x must be 3D [B, N, D]");
    TORCH_CHECK(spec_gate.sizes() == x.sizes(), "spec_gate shape mismatch");
    TORCH_CHECK(decay.sizes() == x.sizes(), "decay shape mismatch");

    const int B = x.size(0);
    const int N = x.size(1);
    const int D = x.size(2);
    auto out = torch::empty_like(x);

    if (B == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto g_c = spec_gate.contiguous();
    auto d_c = decay.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "vspec_fwd", ([&] {
            neuralops::cuda::vspec_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                d_c.data_ptr<scalar_t>(),
                B, N, D, static_cast<float>(eps)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
