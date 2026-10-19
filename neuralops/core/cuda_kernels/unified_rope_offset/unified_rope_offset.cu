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
__global__ void __launch_bounds__(128) unified_rope_offset_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ x,
    const float* __restrict__ cos_tab,
    const float* __restrict__ sin_tab,
    const int B,
    const int H,
    const int N,
    const int D,
    const int half_rope,
    const int offset
) {
    const int bhn = blockIdx.y;
    if (bhn >= B * H * N) return;

    const int n = bhn % N;
    const int pos = n + offset;
    const int base_idx = bhn * D;
    const int tab_base = pos * half_rope;

    const int num_vec = half_rope / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int vi = tid; vi < num_vec; vi += blockDim.x * gridDim.x) {
        const int i = vi * 4;
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 c4 = *reinterpret_cast<const float4*>(cos_tab + tab_base + i);
            const float4 s4 = *reinterpret_cast<const float4*>(sin_tab + tab_base + i);

            const float4 x0_4 = *reinterpret_cast<const float4*>(x + base_idx + i);
            const float4 x1_4 = *reinterpret_cast<const float4*>(x + base_idx + half_rope + i);

            float4 r0_4;
            float4 r1_4;

            #define ROTATE_ROPE(x0, x1, c, s, r0, r1) { \
                r0 = fmaf(x0, c, -(x1 * s)); \
                r1 = fmaf(x0, s, x1 * c); \
            }
            ROTATE_ROPE(x0_4.x, x1_4.x, c4.x, s4.x, r0_4.x, r1_4.x);
            ROTATE_ROPE(x0_4.y, x1_4.y, c4.y, s4.y, r0_4.y, r1_4.y);
            ROTATE_ROPE(x0_4.z, x1_4.z, c4.z, s4.z, r0_4.z, r1_4.z);
            ROTATE_ROPE(x0_4.w, x1_4.w, c4.w, s4.w, r0_4.w, r1_4.w);
            #undef ROTATE_ROPE

            *reinterpret_cast<float4*>(out + base_idx + i) = r0_4;
            *reinterpret_cast<float4*>(out + base_idx + half_rope + i) = r1_4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int curr_i = i + k;
                const float c = cos_tab[tab_base + curr_i];
                const float s = sin_tab[tab_base + curr_i];
                const float x0 = to_float(x[base_idx + curr_i]);
                const float x1 = to_float(x[base_idx + half_rope + curr_i]);

                const float r0 = fmaf(x0, c, -(x1 * s));
                const float r1 = fmaf(x0, s, x1 * c);

                out[base_idx + curr_i] = from_float<scalar_t>(r0);
                out[base_idx + half_rope + curr_i] = from_float<scalar_t>(r1);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < half_rope; i += blockDim.x * gridDim.x) {
        const float c = cos_tab[tab_base + i];
        const float s = sin_tab[tab_base + i];
        const float x0 = to_float(x[base_idx + i]);
        const float x1 = to_float(x[base_idx + half_rope + i]);

        const float r0 = fmaf(x0, c, -(x1 * s));
        const float r1 = fmaf(x0, s, x1 * c);

        out[base_idx + i] = from_float<scalar_t>(r0);
        out[base_idx + half_rope + i] = from_float<scalar_t>(r1);
    }

    const int rope_dim = half_rope * 2;
    for (int i = rope_dim + tid; i < D; i += blockDim.x * gridDim.x) {
        out[base_idx + i] = x[base_idx + i];
    }
}

}}

torch::Tensor unified_rope_offset_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos_tab,
    const torch::Tensor& sin_tab,
    int rope_dim,
    int offset = 0
) {
    CHECK_INPUT(x);
    CHECK_INPUT(cos_tab);
    CHECK_INPUT(sin_tab);

    TORCH_CHECK(x.dim() == 4, "x must be 4D [B, H, N, D]");
    TORCH_CHECK(rope_dim % 2 == 0, "rope_dim must be even");
    TORCH_CHECK(rope_dim <= x.size(3), "rope_dim must not exceed D");

    const int B = x.size(0);
    const int H = x.size(1);
    const int N = x.size(2);
    const int D = x.size(3);
    const int half_rope = rope_dim / 2;

    const int max_pos = cos_tab.size(0);
    TORCH_CHECK(N + offset <= max_pos,
                "unified_rope_offset_fwd: N + offset = ", N + offset,
                " exceeds cos_tab.size(0) = ", max_pos);

    auto out = torch::empty_like(x);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    const int vec_count = (half_rope + 3) / 4;
    dim3 grid((vec_count + THREADS - 1) / THREADS, B * H * N);
    if (grid.x == 0) grid.x = 1;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto x_c = x.contiguous();
    auto cos_c = cos_tab.contiguous().to(torch::kFloat32);
    auto sin_c = sin_tab.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x.scalar_type(), "unified_rope_offset_fwd", ([&] {
            neuralops::cuda::unified_rope_offset_vectorized_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                cos_c.data_ptr<float>(),
                sin_c.data_ptr<float>(),
                B, H, N, D, half_rope, offset
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
