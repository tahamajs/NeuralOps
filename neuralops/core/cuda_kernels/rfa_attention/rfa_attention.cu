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
__global__ void __launch_bounds__(128) rfa_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ phi_q,
    const scalar_t* __restrict__ phi_k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int head_offset = bh * (N * D) + d;

    float s = 0.0f;
    float denom = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float k_val = to_float(phi_k[idx]);
        const float v_val = to_float(v[idx]);
        s = fmaf(k_val, v_val, s);
        denom += k_val;

        const float q_val = to_float(phi_q[idx]);
        const float norm = fmaxf(fabsf(denom), 1e-5f);
        out[idx] = from_float<scalar_t>((q_val * s) / norm);
    }
}

}}

torch::Tensor rfa_fwd(
    const torch::Tensor& phi_q,
    const torch::Tensor& phi_k,
    const torch::Tensor& v
) {
    CHECK_INPUT(phi_q);
    CHECK_INPUT(phi_k);
    CHECK_INPUT(v);

    TORCH_CHECK(phi_q.dim() == 4, "phi_q must be 4D [B, H, N, D]");
    TORCH_CHECK(phi_k.sizes() == phi_q.sizes(), "phi_k shape mismatch");
    TORCH_CHECK(v.sizes() == phi_q.sizes(), "v shape mismatch");

    const int B = phi_q.size(0);
    const int H = phi_q.size(1);
    const int N = phi_q.size(2);
    const int D = phi_q.size(3);

    auto out = torch::empty_like(phi_q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = phi_q.contiguous();
    auto k_c = phi_k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        phi_q.scalar_type(), "rfa_fwd", ([&] {
            neuralops::cuda::rfa_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
