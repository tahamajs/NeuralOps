// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <math.h>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t, int ITEMS_PER_THREAD>
__launch_bounds__(128)
__global__ void linformer_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K_proj,
    const scalar_t* __restrict__ V_proj,
    const int B,
    const int H,
    const int N,
    const int K_proj_len,
    const int D,
    const float scale
) {
    const int warp_id = blockIdx.x * 4 + (threadIdx.x / 32);
    const int lane_id = threadIdx.x % 32;
    const int total = B * H * N;
    if (warp_id >= total) return;

    const int b = warp_id / (H * N);
    const int rem = warp_id % (H * N);
    const int h = rem / N;
    const int i = rem % N;

    const int q_offset = b * (H * N * D) + h * (N * D) + i * D;
    const int kv_head_offset = b * (H * K_proj_len * D) + h * (K_proj_len * D);

    float q_reg[ITEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        int d = lane_id + k * 32;
        q_reg[k] = (d < D) ? to_float(Q[q_offset + d]) : 0.0f;
    }

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc[ITEMS_PER_THREAD];
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        acc[k] = 0.0f;
    }

    for (int j = 0; j < K_proj_len; ++j) {
        const int kv_offset = kv_head_offset + j * D;
        float local_dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            int d = lane_id + k * 32;
            if (d < D) {
                local_dot += q_reg[k] * to_float(K_proj[kv_offset + d]);
            }
        }
        float score = warp_reduce_sum(local_dot) * scale;

        if (score > max_s) {
            const float rescale = expf(max_s - score);
            sum_exp = sum_exp * rescale + 1.0f;
            #pragma unroll
            for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                int d = lane_id + k * 32;
                float v_val = (d < D) ? to_float(V_proj[kv_offset + d]) : 0.0f;
                acc[k] = acc[k] * rescale + v_val;
            }
            max_s = score;
        } else {
            const float weight = expf(score - max_s);
            sum_exp += weight;
            #pragma unroll
            for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                int d = lane_id + k * 32;
                float v_val = (d < D) ? to_float(V_proj[kv_offset + d]) : 0.0f;
                acc[k] += weight * v_val;
            }
        }
    }

    const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
    const int out_offset = b * (H * N * D) + h * (N * D) + i * D;
    #pragma unroll
    for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
        int d = lane_id + k * 32;
        if (d < D) {
            out[out_offset + d] = from_float<scalar_t>(acc[k] * inv_sum);
        }
    }
}

}
}

torch::Tensor linformer_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k_proj,
    const torch::Tensor& v_proj
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k_proj.is_cuda(), "k_proj must be a CUDA tensor");
    TORCH_CHECK(v_proj.is_cuda(), "v_proj must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int K_proj_len = k_proj.size(2);
    TORCH_CHECK(D <= 128, "linformer_attn_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int total = B * H * N;
    const int threads = 128;
    const int blocks = (total + 3) / 4;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k_proj.contiguous();
    auto v_c = v_proj.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "linformer_attn_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::linformer_attention_fwd_kernel<scalar_t, 1><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N, K_proj_len, D, scale
                );
            } else if (D <= 64) {
                neuralops::cuda::linformer_attention_fwd_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N, K_proj_len, D, scale
                );
            } else {
                neuralops::cuda::linformer_attention_fwd_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    B, H, N, K_proj_len, D, scale
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
