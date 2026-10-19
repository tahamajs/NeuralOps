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
__global__ void __launch_bounds__(128, 2) cluster_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ centroids,
    const scalar_t* __restrict__ v,
    const int* __restrict__ cluster_indices,
    const int B,
    const int H,
    const int N,
    const int D,
    const int K
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int q_offset = bh * (N * D) + i * D;
    const int c_base_offset = bh * (K * D);
    const float scale = rsqrtf(static_cast<float>(D));

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (i < N && d < D) {
            q_reg[e] = to_float(q[q_offset + d]) * scale;
        } else {
            q_reg[e] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_c[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int tid = threadIdx.x;

    for (int tile_start = 0; tile_start < K; tile_start += TILE_K) {
        __syncthreads();

        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += 128) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_c = tile_start + t;

                if (global_c < K) {
                    const int g_off = c_base_offset + global_c * D + d_base;
                    const uint4 raw_c = reinterpret_cast<const uint4*>(centroids + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(v + g_off)[0];

                    const scalar_t* p_c = reinterpret_cast<const scalar_t*>(&raw_c);
                    const scalar_t* p_v = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_c[t][d_base + elem] = to_float(p_c[elem]);
                        s_v[t][d_base + elem] = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_c[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else if (sizeof(scalar_t) == 4 && (D % 4 == 0)) {
            const int float4_per_tok = D / 4;
            const int total_float4 = TILE_K * float4_per_tok;
            for (int idx = tid; idx < total_float4; idx += 128) {
                const int t = idx / float4_per_tok;
                const int chunk = idx % float4_per_tok;
                const int d_base = chunk * 4;
                const int global_c = tile_start + t;

                if (global_c < K) {
                    const int g_off = c_base_offset + global_c * D + d_base;
                    const float4 raw_c = reinterpret_cast<const float4*>(centroids + g_off)[0];
                    const float4 raw_v = reinterpret_cast<const float4*>(v + g_off)[0];

                    s_c[t][d_base + 0] = raw_c.x;
                    s_c[t][d_base + 1] = raw_c.y;
                    s_c[t][d_base + 2] = raw_c.z;
                    s_c[t][d_base + 3] = raw_c.w;

                    s_v[t][d_base + 0] = raw_v.x;
                    s_v[t][d_base + 1] = raw_v.y;
                    s_v[t][d_base + 2] = raw_v.z;
                    s_v[t][d_base + 3] = raw_v.w;
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 4; ++elem) {
                        s_c[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            const int tile_elements = TILE_K * D;
            for (int idx = tid; idx < tile_elements; idx += 128) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_c = tile_start + t;
                if (global_c < K) {
                    const int g_off = c_base_offset + global_c * D + d;
                    s_c[t][d] = to_float(centroids[g_off]);
                    s_v[t][d] = to_float(v[g_off]);
                } else {
                    s_c[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N) {
            const int max_t = min(TILE_K, K - tile_start);
            for (int t = 0; t < max_t; ++t) {
                float dot_local = 0.0f;
                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot_local = fmaf(q_reg[e], s_c[t][d], dot_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                }

                const float m_curr = fmaxf(m_prev, dot_local);
                const float alpha = expf(m_prev - m_curr);
                const float p = expf(dot_local - m_curr);
                d_prev = d_prev * alpha + p;
                m_prev = m_curr;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        const float v_val = s_v[t][d];
                        acc[e] = acc[e] * alpha + p * v_val;
                    }
                }
            }
        }
    }

    if (i < N) {
        const float inv_d = (d_prev > 1e-6f) ? (1.0f / d_prev) : 0.0f;
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                out[q_offset + d] = from_float<scalar_t>(acc[e] * inv_d);
            }
        }
    }
}

}}

torch::Tensor cluster_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& cluster_centroids,
    const torch::Tensor& v,
    const torch::Tensor& cluster_indices
) {
    CHECK_INPUT(q);
    CHECK_INPUT(cluster_centroids);
    CHECK_INPUT(v);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(cluster_centroids.dim() == 4, "cluster_centroids must be 4D [B, H, K, D]");
    TORCH_CHECK(v.dim() == 4, "v must be 4D [B, H, K, D]");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int K = cluster_centroids.size(2);
    TORCH_CHECK(D <= 128, "cluster_attn_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0 || K == 0) return out;

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto c_c = cluster_centroids.contiguous();
    auto v_c = v.contiguous();
    auto idx_c = cluster_indices.defined() ? cluster_indices.contiguous() : torch::empty(0);

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cluster_attn_fwd", ([&] {
            neuralops::cuda::cluster_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                c_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                idx_c.defined() ? idx_c.data_ptr<int>() : nullptr,
                B, H, N, D, K
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
