// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <cmath>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

__device__ __forceinline__ float elu_plus_one(float x) {
    return (x > 0.0f) ? (x + 1.0f) : expf(x);
}

// Optimized Chunk-Parallel Linear Attention for Cosformer (FLA Paradigm)
// Eliminates O(L^2) repeated DRAM reads by accumulating S_cos and S_sin state across chunks in registers.
template <typename scalar_t, int MAX_D>
__launch_bounds__(128)
__global__ void cosformer_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int L,
    const int D,
    const bool is_causal
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    if (b >= B || h >= H) return;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;

    // Registers / Shared memory for decoupled S_cos and S_sin state accumulators
    // MAX_D + 1 padding eliminates shared memory bank conflicts across all 32 banks
    __shared__ float s_S_cos[MAX_D][MAX_D + 1];
    __shared__ float s_S_sin[MAX_D][MAX_D + 1];
    __shared__ float s_Z_cos[MAX_D];
    __shared__ float s_Z_sin[MAX_D];

    __shared__ float s_k_cos[MAX_D];
    __shared__ float s_k_sin[MAX_D];
    __shared__ float s_v[MAX_D];
    __shared__ float s_q_cos[MAX_D];
    __shared__ float s_q_sin[MAX_D];

    // Zero out inter-chunk accumulators
    for (int idx = tid; idx < D * D; idx += num_threads) {
        int r = idx / D;
        int c = idx % D;
        s_S_cos[r][c] = 0.0f;
        s_S_sin[r][c] = 0.0f;
    }
    if (tid < D) {
        s_Z_cos[tid] = 0.0f;
        s_Z_sin[tid] = 0.0f;
    }
    __syncthreads();

    const int base_bh = (b * H + h) * L;
    const float theta_scale = 3.141592653589793f / (2.0f * (float)L);

    if (is_causal) {
        // Sequential scan over L tokens maintaining state S_cos, S_sin in SRAM (O(L) memory reads)
        for (int t = 0; t < L; ++t) {
            const int offset = (base_bh + t) * D;
            const float theta = theta_scale * (float)t;
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            if (tid < D) {
                const float raw_k = to_float(k[offset + tid]);
                const float phi_k = elu_plus_one(raw_k);
                s_k_cos[tid] = phi_k * cos_theta;
                s_k_sin[tid] = phi_k * sin_theta;

                const float raw_q = to_float(q[offset + tid]);
                const float phi_q = elu_plus_one(raw_q);
                s_q_cos[tid] = phi_q * cos_theta;
                s_q_sin[tid] = phi_q * sin_theta;

                s_v[tid] = to_float(v[offset + tid]);

                s_Z_cos[tid] += s_k_cos[tid];
                s_Z_sin[tid] += s_k_sin[tid];
            }
            __syncthreads();

            // Rank-1 state update: S_cos += k_cos * v^T, S_sin += k_sin * v^T
            for (int idx = tid; idx < D * D; idx += num_threads) {
                int r = idx / D;
                int c = idx % D;
                s_S_cos[r][c] += s_k_cos[r] * s_v[c];
                s_S_sin[r][c] += s_k_sin[r] * s_v[c];
            }
            __syncthreads();

            // Compute causal output for token t:
            // num = q_cos * S_cos + q_sin * S_sin
            // denom = q_cos * Z_cos + q_sin * Z_sin
            float local_denom = 0.0f;
            for (int i = tid; i < D; i += num_threads) {
                local_denom += s_q_cos[i] * s_Z_cos[i] + s_q_sin[i] * s_Z_sin[i];
            }
            local_denom = block_reduce_sum(local_denom);
            __shared__ float s_denom;
            if (tid == 0) {
                s_denom = fmaxf(local_denom, 1e-5f);
            }
            __syncthreads();
            const float denom = s_denom;

            for (int col = tid; col < D; col += num_threads) {
                float num = 0.0f;
                for (int r = 0; r < D; ++r) {
                    num += s_q_cos[r] * s_S_cos[r][col] + s_q_sin[r] * s_S_sin[r][col];
                }
                out[offset + col] = from_float<scalar_t>(num / denom);
            }
            __syncthreads();
        }
    } else {
        // Non-causal: Pass 1 accumulate S_cos, S_sin, Z_cos, Z_sin over all tokens
        for (int t = 0; t < L; ++t) {
            const int offset = (base_bh + t) * D;
            const float theta = theta_scale * (float)t;
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            if (tid < D) {
                const float raw_k = to_float(k[offset + tid]);
                const float phi_k = elu_plus_one(raw_k);
                s_k_cos[tid] = phi_k * cos_theta;
                s_k_sin[tid] = phi_k * sin_theta;

                s_v[tid] = to_float(v[offset + tid]);

                s_Z_cos[tid] += s_k_cos[tid];
                s_Z_sin[tid] += s_k_sin[tid];
            }
            __syncthreads();

            for (int idx = tid; idx < D * D; idx += num_threads) {
                int r = idx / D;
                int c = idx % D;
                s_S_cos[r][c] += s_k_cos[r] * s_v[c];
                s_S_sin[r][c] += s_k_sin[r] * s_v[c];
            }
            __syncthreads();
        }

        // Pass 2: Compute out for each token t
        for (int t = 0; t < L; ++t) {
            const int offset = (base_bh + t) * D;
            const float theta = theta_scale * (float)t;
            const float cos_theta = cosf(theta);
            const float sin_theta = sinf(theta);

            if (tid < D) {
                const float raw_q = to_float(q[offset + tid]);
                const float phi_q = elu_plus_one(raw_q);
                s_q_cos[tid] = phi_q * cos_theta;
                s_q_sin[tid] = phi_q * sin_theta;
            }
            __syncthreads();

            float local_denom = 0.0f;
            for (int i = tid; i < D; i += num_threads) {
                local_denom += s_q_cos[i] * s_Z_cos[i] + s_q_sin[i] * s_Z_sin[i];
            }
            local_denom = block_reduce_sum(local_denom);
            __shared__ float s_denom;
            if (tid == 0) {
                s_denom = fmaxf(local_denom, 1e-5f);
            }
            __syncthreads();
            const float denom = s_denom;

            for (int col = tid; col < D; col += num_threads) {
                float num = 0.0f;
                for (int r = 0; r < D; ++r) {
                    num += s_q_cos[r] * s_S_cos[r][col] + s_q_sin[r] * s_S_sin[r][col];
                }
                out[offset + col] = from_float<scalar_t>(num / denom);
            }
            __syncthreads();
        }
    }
}

}
}

torch::Tensor cosformer_linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const bool is_causal
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);

    const int B = q.size(0);
    const int H = q.size(1);
    const int L = q.size(2);
    const int D = q.size(3);
    TORCH_CHECK(D <= 128, "cosformer_linear_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(v);
    const dim3 grid(B, H);
    const int threads = (D <= 32) ? 32 : ((D <= 64) ? 64 : 128);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cosformer_linear_attention_fwd", ([&] {
            if (D <= 32) {
                neuralops::cuda::cosformer_fwd_kernel<scalar_t, 32><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), B, H, L, D, is_causal
                );
            } else if (D <= 64) {
                neuralops::cuda::cosformer_fwd_kernel<scalar_t, 64><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), B, H, L, D, is_causal
                );
            } else {
                neuralops::cuda::cosformer_fwd_kernel<scalar_t, 128><<<grid, threads, 0, stream>>>(
                    out.data_ptr<scalar_t>(), q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(), B, H, L, D, is_causal
                );
            }
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
