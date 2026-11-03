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

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128) hgrn2_bidirectional_scan_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ g,
    const int B,
    const int H,
    const int L,
    const int D
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    // Vectorized 128-bit float4 path for float type when D is divisible by 4
    if constexpr (std::is_same_v<scalar_t, float>) {
        if (D % 4 == 0) {
            const int num_vec = D / 4;
            const int vi = blockIdx.x * blockDim.x + threadIdx.x;
            if (vi >= num_vec) return;

            const int base_bh = bh * (L * D) + vi * 4;
            float4 h_fwd = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

            #pragma unroll 4
            for (int t = 0; t < L; ++t) {
                const int idx = base_bh + t * D;
                const float4 x_t = *reinterpret_cast<const float4*>(x + idx);
                const float4 g_t = *reinterpret_cast<const float4*>(g + idx);

                h_fwd.x = fmaf(g_t.x, h_fwd.x, (1.0f - g_t.x) * x_t.x);
                h_fwd.y = fmaf(g_t.y, h_fwd.y, (1.0f - g_t.y) * x_t.y);
                h_fwd.z = fmaf(g_t.z, h_fwd.z, (1.0f - g_t.z) * x_t.z);
                h_fwd.w = fmaf(g_t.w, h_fwd.w, (1.0f - g_t.w) * x_t.w);

                *reinterpret_cast<float4*>(out + idx) = h_fwd;
            }

            float4 h_bwd = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll 4
            for (int t = L - 1; t >= 0; --t) {
                const int idx = base_bh + t * D;
                const float4 x_t = *reinterpret_cast<const float4*>(x + idx);
                const float4 g_t = *reinterpret_cast<const float4*>(g + idx);

                h_bwd.x = fmaf(g_t.x, h_bwd.x, (1.0f - g_t.x) * x_t.x);
                h_bwd.y = fmaf(g_t.y, h_bwd.y, (1.0f - g_t.y) * x_t.y);
                h_bwd.z = fmaf(g_t.z, h_bwd.z, (1.0f - g_t.z) * x_t.z);
                h_bwd.w = fmaf(g_t.w, h_bwd.w, (1.0f - g_t.w) * x_t.w);

                const float4 fwd_val = *reinterpret_cast<const float4*>(out + idx);
                float4 combined;
                combined.x = 0.5f * (fwd_val.x + h_bwd.x);
                combined.y = 0.5f * (fwd_val.y + h_bwd.y);
                combined.z = 0.5f * (fwd_val.z + h_bwd.z);
                combined.w = 0.5f * (fwd_val.w + h_bwd.w);

                *reinterpret_cast<float4*>(out + idx) = combined;
            }
            return;
        }
    }

    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= D) return;

    const int base_bh = bh * (L * D) + d;

    float h_fwd = 0.0f;
    #pragma unroll 4
    for (int t = 0; t < L; ++t) {
        const int idx = base_bh + t * D;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_fwd = fmaf(g_t, h_fwd, (1.0f - g_t) * x_t);
        out[idx] = from_float<scalar_t>(h_fwd);
    }

    float h_bwd = 0.0f;
    #pragma unroll 4
    for (int t = L - 1; t >= 0; --t) {
        const int idx = base_bh + t * D;
        const float x_t = to_float(x[idx]);
        const float g_t = to_float(g[idx]);

        h_bwd = fmaf(g_t, h_bwd, (1.0f - g_t) * x_t);
        const float fwd_val = to_float(out[idx]);
        const float combined = 0.5f * (fwd_val + h_bwd);
        out[idx] = from_float<scalar_t>(combined);
    }
}

}
}

torch::Tensor hgrn2_bidirectional_scan_fwd(
    const torch::Tensor& x,
    const torch::Tensor& g
) {
    CHECK_INPUT(x);
    CHECK_INPUT(g);

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, L, D]");
    TORCH_CHECK(g.sizes() == x.sizes(), "g shape mismatch");

    const int B = x.size(0);
    const int H = x.size(1);
    const int L = x.size(2);
    const int D = x.size(3);

    auto out = torch::empty_like(x);
    if (B == 0 || H == 0 || L == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int blocks_x = (D + THREADS - 1) / THREADS;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto g_c = g.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "hgrn2_bidirectional_scan_fwd", ([&] {
            neuralops::cuda::hgrn2_bidirectional_scan_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                B, H, L, D
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
