// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(256) recurrent_hgrn2_step_vectorized_kernel(
    scalar_t* __restrict__ out,
    scalar_t* __restrict__ h_state,
    const scalar_t* __restrict__ x_t,
    const scalar_t* __restrict__ g_t,
    const scalar_t* __restrict__ alpha_t,
    const int total_elements
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 a4 = *reinterpret_cast<const float4*>(alpha_t + idx * 4);
            const float4 x4 = *reinterpret_cast<const float4*>(x_t + idx * 4);
            const float4 h4 = *reinterpret_cast<const float4*>(h_state + idx * 4);
            const float4 g4 = *reinterpret_cast<const float4*>(g_t + idx * 4);

            float4 h_new4;
            float4 out4;

            #define STEP_HGRN2(a_v, x_v, h_v, g_v, h_n, o_v) { \
                h_n = fmaf(a_v, (h_v) - (x_v), x_v); \
                o_v = (h_n) * (g_v); \
            }
            STEP_HGRN2(a4.x, x4.x, h4.x, g4.x, h_new4.x, out4.x);
            STEP_HGRN2(a4.y, x4.y, h4.y, g4.y, h_new4.y, out4.y);
            STEP_HGRN2(a4.z, x4.z, h4.z, g4.z, h_new4.z, out4.z);
            STEP_HGRN2(a4.w, x4.w, h4.w, g4.w, h_new4.w, out4.w);
            #undef STEP_HGRN2

            *reinterpret_cast<float4*>(h_state + idx * 4) = h_new4;
            *reinterpret_cast<float4*>(out + idx * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = idx * 4 + k;
                const float a_val = to_float(alpha_t[elem_idx]);
                const float x_val = to_float(x_t[elem_idx]);
                const float h_val = to_float(h_state[elem_idx]);
                const float g_val = to_float(g_t[elem_idx]);
                const float h_new = fmaf(a_val, h_val - x_val, x_val);
                h_state[elem_idx] = from_float<scalar_t>(h_new);
                out[elem_idx] = from_float<scalar_t>(h_new * g_val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float a_val = to_float(alpha_t[idx]);
        const float x_val = to_float(x_t[idx]);
        const float h_val = to_float(h_state[idx]);
        const float g_val = to_float(g_t[idx]);

        const float h_new = fmaf(a_val, h_val - x_val, x_val);
        h_state[idx] = from_float<scalar_t>(h_new);
        out[idx] = from_float<scalar_t>(h_new * g_val);
    }
}

}
}

std::vector<torch::Tensor> recurrent_hgrn2_step_fwd(
    const torch::Tensor& x_t,
    const torch::Tensor& g_t,
    const torch::Tensor& alpha_t,
    torch::Tensor& h_prev
) {
    CHECK_INPUT(x_t);
    CHECK_INPUT(g_t);
    CHECK_INPUT(alpha_t);
    CHECK_INPUT(h_prev);

    auto out = torch::empty_like(x_t);
    const int total = x_t.numel();
    if (total == 0) return {out, h_prev};

    const int vec_count = (total + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto x_c = x_t.contiguous();
    auto g_c = g_t.contiguous();
    auto a_c = alpha_t.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        x_t.scalar_type(), "recurrent_hgrn2_step_fwd", ([&] {
            neuralops::cuda::recurrent_hgrn2_step_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                h_prev.data_ptr<scalar_t>(),
                x_c.data_ptr<scalar_t>(),
                g_c.data_ptr<scalar_t>(),
                a_c.data_ptr<scalar_t>(),
                total
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {out, h_prev};
}
