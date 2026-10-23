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
__global__ void __launch_bounds__(128) rwkv6_time_mix_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ r,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ u,
    const int B,
    const int H,
    const int N,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int h = bh % H;
    const float u_val = to_float(u[h * D + d]);
    const int head_offset = bh * (N * D) + d;

    float s = 0.0f;

    #pragma unroll 4
    for (int t = 0; t < N; ++t) {
        const int idx = head_offset + t * D;
        const float r_raw = to_float(r[idx]);
        const float r_val = 1.0f / (1.0f + expf(-r_raw));
        const float k_val = expf(to_float(k[idx]));
        const float v_val = to_float(v[idx]);
        const float decay = expf(-expf(to_float(w[idx])));

        const float kv = k_val * v_val;
        const float y = r_val * (u_val * kv + s);
        s = fmaf(decay, s, kv);

        out[idx] = from_float<scalar_t>(y);
    }
}

}}

torch::Tensor rwkv6_time_mix_fwd(
    const torch::Tensor& r,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& w,
    const torch::Tensor& u
) {
    CHECK_INPUT(r);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(w);
    CHECK_INPUT(u);

    TORCH_CHECK(r.dim() == 4, "r must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == r.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == r.sizes(), "v shape mismatch");
    TORCH_CHECK(w.sizes() == r.sizes(), "w shape mismatch");

    const int B = r.size(0);
    const int H = r.size(1);
    const int N = r.size(2);
    const int D = r.size(3);

    auto out = torch::empty_like(r);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto r_c = r.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto w_c = w.contiguous();
    auto u_c = u.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        r.scalar_type(), "rwkv6_time_mix_fwd", ([&] {
            neuralops::cuda::rwkv6_time_mix_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                r_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                w_c.data_ptr<scalar_t>(),
                u_c.data_ptr<scalar_t>(),
                B, H, N, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
