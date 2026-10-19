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
__global__ void __launch_bounds__(512) aegis_gate_norm_2b_vectorized_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ b0,
    const scalar_t* __restrict__ b1,
    const float* __restrict__ weights,
    const int total_tokens,
    const int D,
    const float eps
) {
    const int pid = blockIdx.x;
    if (pid >= total_tokens) return;
    const int tid = threadIdx.x;

    const float raw_w0 = weights[pid * 2 + 0];
    const float raw_w1 = weights[pid * 2 + 1];
    const float max_w = fmaxf(raw_w0, raw_w1);
    const float exp0 = expf(raw_w0 - max_w);
    const float exp1 = expf(raw_w1 - max_w);
    const float inv_sum = 1.0f / (exp0 + exp1 + 1e-8f);
    const float w0 = exp0 * inv_sum;
    const float w1 = exp1 * inv_sum;

    const int base_idx = pid * D;
    const int num_vec = D / 4;

    float sq0 = 0.0f, sq1 = 0.0f;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 v0_4 = *reinterpret_cast<const float4*>(b0 + base_idx + i * 4);
            const float4 v1_4 = *reinterpret_cast<const float4*>(b1 + base_idx + i * 4);

            sq0 = fmaf(v0_4.x, v0_4.x, sq0);
            sq0 = fmaf(v0_4.y, v0_4.y, sq0);
            sq0 = fmaf(v0_4.z, v0_4.z, sq0);
            sq0 = fmaf(v0_4.w, v0_4.w, sq0);

            sq1 = fmaf(v1_4.x, v1_4.x, sq1);
            sq1 = fmaf(v1_4.y, v1_4.y, sq1);
            sq1 = fmaf(v1_4.z, v1_4.z, sq1);
            sq1 = fmaf(v1_4.w, v1_4.w, sq1);
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float val0 = to_float(b0[base_idx + i * 4 + k]);
                const float val1 = to_float(b1[base_idx + i * 4 + k]);
                sq0 = fmaf(val0, val0, sq0);
                sq1 = fmaf(val1, val1, sq1);
            }
        }
    }

    const int rem_start = num_vec * 4;
    for (int i = rem_start + tid; i < D; i += blockDim.x) {
        const float val0 = to_float(b0[base_idx + i]);
        const float val1 = to_float(b1[base_idx + i]);
        sq0 = fmaf(val0, val0, sq0);
        sq1 = fmaf(val1, val1, sq1);
    }

    sq0 = block_reduce_sum(sq0);
    sq1 = block_reduce_sum(sq1);

    __shared__ float s_inv0, s_inv1;
    if (tid == 0) {
        s_inv0 = rsqrtf(sq0 / (float)D + eps);
        s_inv1 = rsqrtf(sq1 / (float)D + eps);
    }
    __syncthreads();

    const float inv_rms0 = s_inv0;
    const float inv_rms1 = s_inv1;

    for (int i = tid; i < num_vec; i += blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, float>) {
            const float4 v0_4 = *reinterpret_cast<const float4*>(b0 + base_idx + i * 4);
            const float4 v1_4 = *reinterpret_cast<const float4*>(b1 + base_idx + i * 4);

            float4 out4;
            out4.x = v0_4.x * inv_rms0 * w0 + v1_4.x * inv_rms1 * w1;
            out4.y = v0_4.y * inv_rms0 * w0 + v1_4.y * inv_rms1 * w1;
            out4.z = v0_4.z * inv_rms0 * w0 + v1_4.z * inv_rms1 * w1;
            out4.w = v0_4.w * inv_rms0 * w0 + v1_4.w * inv_rms1 * w1;

            *reinterpret_cast<float4*>(out + base_idx + i * 4) = out4;
        } else {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int elem_idx = base_idx + i * 4 + k;
                const float n0 = to_float(b0[elem_idx]) * inv_rms0;
                const float n1 = to_float(b1[elem_idx]) * inv_rms1;
                out[elem_idx] = from_float<scalar_t>(n0 * w0 + n1 * w1);
            }
        }
    }

    for (int i = rem_start + tid; i < D; i += blockDim.x) {
        const int elem_idx = base_idx + i;
        const float n0 = to_float(b0[elem_idx]) * inv_rms0;
        const float n1 = to_float(b1[elem_idx]) * inv_rms1;
        out[elem_idx] = from_float<scalar_t>(n0 * w0 + n1 * w1);
    }
}

}}

torch::Tensor aegis_gate_norm_2branch_fwd(
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    const torch::Tensor& weights,
    float eps
) {
    CHECK_INPUT(b0);
    CHECK_INPUT(b1);
    CHECK_INPUT(weights);

    const int D = b0.size(-1);
    const int total_tokens = b0.numel() / D;
    auto out = torch::empty_like(b0);
    if (total_tokens == 0 || D == 0) return out;

    int threads = std::min(512, ((D + 31) / 32) * 32);
    if (threads < 32) threads = 32;

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto b0_c = b0.contiguous();
    auto b1_c = b1.contiguous();
    auto w_c = weights.contiguous().to(torch::kFloat32);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        b0.scalar_type(), "aegis_gate_norm_2branch_fwd", ([&] {
            neuralops::cuda::aegis_gate_norm_2b_vectorized_kernel<scalar_t><<<total_tokens, threads, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                b0_c.data_ptr<scalar_t>(),
                b1_c.data_ptr<scalar_t>(),
                w_c.data_ptr<float>(),
                total_tokens,
                D,
                eps
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
