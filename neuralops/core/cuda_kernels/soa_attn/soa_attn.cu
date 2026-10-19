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
__global__ void __launch_bounds__(256) soa_attn_fwd_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int total_elements,
    const float alpha
) {
    const int num_vec = total_elements / 4;
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;

    for (int idx = tid; idx < num_vec; idx += stride) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 q4 = *reinterpret_cast<const float4*>(q + idx * 4);
            const float4 k4 = *reinterpret_cast<const float4*>(k + idx * 4);
            const float4 v4 = *reinterpret_cast<const float4*>(v + idx * 4);
            float4 out4;

            #define COMPUTE_SOA(q_elem, k_elem, v_elem, out_elem) { \
                const float qk = (q_elem) * (k_elem); \
                const float phi = fmaf(alpha, qk * qk, qk); \
                (out_elem) = phi * (v_elem); \
            }
            COMPUTE_SOA(q4.x, k4.x, v4.x, out4.x);
            COMPUTE_SOA(q4.y, k4.y, v4.y, out4.y);
            COMPUTE_SOA(q4.z, k4.z, v4.z, out4.z);
            COMPUTE_SOA(q4.w, k4.w, v4.w, out4.w);
            #undef COMPUTE_SOA

            *reinterpret_cast<float4*>(out + idx * 4) = out4;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int elem_idx = idx * 4 + j;
                const float q_val = to_float(q[elem_idx]);
                const float k_val = to_float(k[elem_idx]);
                const float v_val = to_float(v[elem_idx]);
                const float qk = q_val * k_val;
                const float phi = fmaf(alpha, qk * qk, qk);
                out[elem_idx] = from_float<scalar_t>(phi * v_val);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int idx = rem_start + tid; idx < total_elements; idx += stride) {
        const float q_val = to_float(q[idx]);
        const float k_val = to_float(k[idx]);
        const float v_val = to_float(v[idx]);
        const float qk = q_val * k_val;
        const float phi = fmaf(alpha, qk * qk, qk);
        out[idx] = from_float<scalar_t>(phi * v_val);
    }
}

}}

torch::Tensor soa_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double alpha
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    auto out = torch::empty_like(q);
    const int total = q.numel();
    if (total == 0) return out;

    const int vec_count = (total + 3) / 4;
    constexpr int THREADS = 256;
    const int blocks = std::min(65535, (vec_count + THREADS - 1) / THREADS);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "soa_attn_fwd", ([&] {
            neuralops::cuda::soa_attn_fwd_vectorized_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                total,
                static_cast<float>(alpha)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
