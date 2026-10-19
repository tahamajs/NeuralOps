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
__global__ void __launch_bounds__(128) hydra_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
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

    float kv_sum = 0.0f;
    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);
        kv_sum = fmaf(k_val, v_val, kv_sum);
    }
    const float norm = (N > 0) ? (1.0f / static_cast<float>(N)) : 0.0f;
    const float global_kv = kv_sum * norm;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float q_val = to_float(q[idx]);
        out[idx] = from_float<scalar_t>(q_val * global_kv);
    }
}

}}

torch::Tensor hydra_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "hydra_fwd", ([&] {
            neuralops::cuda::hydra_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
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
