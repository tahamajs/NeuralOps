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

__device__ __forceinline__ float compute_alibi_slope(const int h, const int H) {
    // If H is a power of 2: slope_h = 2^(-8 * (h + 1) / H)
    // If H is not a power of 2, standard Press et al. ALiBi formulation:
    // let n = 2^(floor(log2(H))). For h < n: 2^(-8 * (h + 1) / n)
    // For h >= n: 2^(-8 * (2 * (h - n) + 1) / (2 * n))
    // Optimized single-branch in-register evaluation:
    const int n = 1 << (31 - __clz(H));
    if (h < n) {
        return __exp2f(-8.0f * static_cast<float>(h + 1) / static_cast<float>(n));
    } else {
        return __exp2f(-8.0f * static_cast<float>(2 * (h - n) + 1) / static_cast<float>(2 * n));
    }
}

template <typename scalar_t, int ITEMS_PER_THREAD>
__launch_bounds__(128)
__global__ void alibi_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const float* __restrict__ slopes,
    const int B,
    const int H,
    const int L,
    const int D,
    const float scale,
    const bool is_causal
) {
    const int warp_id = blockIdx.x * 4 + (threadIdx.x / 32);
    const int lane_id = threadIdx.x % 32;
    const int total = B * H * L;
    if (warp_id >= total) return;

    const int b = warp_id / (H * L);
    const int rem = warp_id % (H * L);
    const int h = rem / L;
    const int i = rem % L;

    // Phase 1: In-register geometric slope coefficient generation
    // Fall back to slopes pointer only if explicitly provided, else compute directly in registers
    const float slope = (slopes != nullptr) ? slopes[h] : compute_alibi_slope(h, H);
    const int q_offset = b * (H * L * D) + h * (L * D) + i * D;

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

    const int kv_head_offset = b * (H * L * D) + h * (L * D);
    const int j_limit = is_causal ? (i + 1) : L;

    for (int j = 0; j < j_limit; ++j) {
        const int kv_offset = kv_head_offset + j * D;
        float local_dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
            int d = lane_id + k * 32;
            if (d < D) {
                local_dot += q_reg[k] * to_float(K[kv_offset + d]);
            }
        }
        // Phase 2: Online linear position penalty fused directly into softmax accumulator
        const float pos_dist = is_causal ? static_cast<float>(i - j) : fabsf(static_cast<float>(i - j));
        const float score = warp_reduce_sum(local_dot) * scale - slope * pos_dist;

        if (score > max_s) {
            const float rescale = expf(max_s - score);
            sum_exp = sum_exp * rescale + 1.0f;
            #pragma unroll
            for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                int d = lane_id + k * 32;
                float v_val = (d < D) ? to_float(V[kv_offset + d]) : 0.0f;
                acc[k] = acc[k] * rescale + v_val;
            }
            max_s = score;
        } else {
            const float weight = expf(score - max_s);
            sum_exp += weight;
            #pragma unroll
            for (int k = 0; k < ITEMS_PER_THREAD; ++k) {
                int d = lane_id + k * 32;
                float v_val = (d < D) ? to_float(V[kv_offset + d]) : 0.0f;
                acc[k] += weight * v_val;
            }
        }
    }

    const float inv_sum = 1.0f / (sum_exp != 0.0f ? sum_exp : 1.0f);
    const int out_offset = b * (H * L * D) + h * (L * D) + i * D;
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

torch::Tensor alibi_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& slopes,
    const bool is_causal
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);
    TORCH_CHECK(D <= 128, "alibi_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    const int total = B * H * L;
    const int threads = 128;
    const int blocks = (total + 3) / 4;
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    const float* slopes_ptr = nullptr;
    torch::Tensor slopes_c;
    if (slopes.defined() && slopes.numel() > 0) {
        slopes_c = slopes.contiguous().to(torch::kFloat32);
        slopes_ptr = slopes_c.data_ptr<float>();
    }

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "alibi_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::alibi_attention_fwd_kernel<scalar_t, 1><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    slopes_ptr,
                    B, H, L, D, scale, is_causal
                );
            } else if (D <= 64) {
                neuralops::cuda::alibi_attention_fwd_kernel<scalar_t, 2><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    slopes_ptr,
                    B, H, L, D, scale, is_causal
                );
            } else {
                neuralops::cuda::alibi_attention_fwd_kernel<scalar_t, 4><<<blocks, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q_c.data_ptr<scalar_t>(),
                    k_c.data_ptr<scalar_t>(),
                    v_c.data_ptr<scalar_t>(),
                    slopes_ptr,
                    B, H, L, D, scale, is_causal
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
