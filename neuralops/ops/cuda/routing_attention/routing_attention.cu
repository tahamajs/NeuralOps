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

namespace neuralops { namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) routing_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ route_weights,
    const int B,
    const int H,
    const int N,
    const int D,
    const int top_k
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (i < N && d < D) {
            q_reg[e] = to_float(q[qi_offset + d]) * scale;
        } else {
            q_reg[e] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];
    __shared__ float s_rw[TILE_K];

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int tid = threadIdx.x;
    const int total_threads = 128;
    const int tile_elements = TILE_K * D;

    for (int tile_start = 0; tile_start < N; tile_start += TILE_K) {
        __syncthreads();
        if (tid < TILE_K) {
            const int global_j = tile_start + tid;
            s_rw[tid] = (global_j < N) ? to_float(route_weights[bh * N + global_j]) : 0.0f;
        }

        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(k + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(v + g_off)[0];

                    const scalar_t* p_k = reinterpret_cast<const scalar_t*>(&raw_k);
                    const scalar_t* p_v = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = to_float(p_k[elem]);
                        s_v[t][d_base + elem] = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            for (int idx = tid; idx < tile_elements; idx += total_threads) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k[t][d] = to_float(k[g_off]);
                    s_v[t][d] = to_float(v[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N) {
            const int max_t = min(TILE_K, N - tile_start);
            for (int t = 0; t < max_t; ++t) {
                if (s_rw[t] <= 0.0f) continue;

                float dot_local = 0.0f;
                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot_local = fmaf(q_reg[e], s_k[t][d], dot_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                }

                const float m_new = fmaxf(m_prev, dot_local);
                const float alpha = expf(m_prev - m_new);
                const float p = expf(dot_local - m_new);
                d_prev = d_prev * alpha + p;
                m_prev = m_new;

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
                out[qi_offset + d] = from_float<scalar_t>(acc[e] * inv_d);
            }
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) routing_attn_bwd_kernel(
    scalar_t* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k,
    const scalar_t* __restrict__ v,
    const scalar_t* __restrict__ route_weights,
    const int B,
    const int H,
    const int N,
    const int D,
    const int top_k
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];
    float go_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (i < N && d < D) {
            q_reg[e] = to_float(q[qi_offset + d]) * scale;
            go_reg[e] = to_float(grad_out[qi_offset + d]);
        } else {
            q_reg[e] = 0.0f;
            go_reg[e] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];
    __shared__ float s_rw[TILE_K];

    float m_prev = -1e30f;
    float d_prev = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int tid = threadIdx.x;
    const int total_threads = 128;
    const int tile_elements = TILE_K * D;

    for (int tile_start = 0; tile_start < N; tile_start += TILE_K) {
        __syncthreads();
        if (tid < TILE_K) {
            const int global_j = tile_start + tid;
            s_rw[tid] = (global_j < N) ? to_float(route_weights[bh * N + global_j]) : 0.0f;
        }

        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(k + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(v + g_off)[0];

                    const scalar_t* p_k = reinterpret_cast<const scalar_t*>(&raw_k);
                    const scalar_t* p_v = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = to_float(p_k[elem]);
                        s_v[t][d_base + elem] = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            for (int idx = tid; idx < tile_elements; idx += total_threads) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k[t][d] = to_float(k[g_off]);
                    s_v[t][d] = to_float(v[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N) {
            const int max_t = min(TILE_K, N - tile_start);
            for (int t = 0; t < max_t; ++t) {
                if (s_rw[t] <= 0.0f) continue;

                float dot_local = 0.0f;
                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot_local = fmaf(q_reg[e], s_k[t][d], dot_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                }

                const float m_new = fmaxf(m_prev, dot_local);
                const float alpha = expf(m_prev - m_new);
                const float p = expf(dot_local - m_new);
                d_prev = d_prev * alpha + p;
                m_prev = m_new;

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

    float D_val = 0.0f;
    if (i < N && d_prev > 1e-6f) {
        const float inv_d = 1.0f / d_prev;
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float o_d = acc[e] * inv_d;
                D_val += go_reg[e] * o_d;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            D_val += __shfl_xor_sync(0xffffffff, D_val, offset);
        }
    }

    float grad_q_acc[MAX_E] = {0.0f};

    for (int tile_start = 0; tile_start < N; tile_start += TILE_K) {
        __syncthreads();
        if (tid < TILE_K) {
            const int global_j = tile_start + tid;
            s_rw[tid] = (global_j < N) ? to_float(route_weights[bh * N + global_j]) : 0.0f;
        }

        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(k + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(v + g_off)[0];

                    const scalar_t* p_k = reinterpret_cast<const scalar_t*>(&raw_k);
                    const scalar_t* p_v = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = to_float(p_k[elem]);
                        s_v[t][d_base + elem] = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem] = 0.0f;
                    }
                }
            }
        } else {
            for (int idx = tid; idx < tile_elements; idx += total_threads) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k[t][d] = to_float(k[g_off]);
                    s_v[t][d] = to_float(v[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N && d_prev > 1e-6f) {
            const int max_t = min(TILE_K, N - tile_start);
            const float inv_d = 1.0f / d_prev;

            for (int t = 0; t < max_t; ++t) {
                if (s_rw[t] <= 0.0f) continue;

                float dot_local = 0.0f;
                float dp_local = 0.0f;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot_local = fmaf(q_reg[e], s_k[t][d], dot_local);
                        dp_local  = fmaf(go_reg[e], s_v[t][d], dp_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot_local += __shfl_xor_sync(0xffffffff, dot_local, offset);
                    dp_local  += __shfl_xor_sync(0xffffffff, dp_local, offset);
                }

                const float p_ij = expf(dot_local - m_prev) * inv_d;
                const float dS_ij = p_ij * (dp_local - D_val) * scale;

                const int global_j = tile_start + t;
                const int kj_offset = base_offset + global_j * D;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        grad_q_acc[e] += dS_ij * s_k[t][d];

                        const float gk_val = dS_ij * to_float(q[qi_offset + d]);
                        const float gv_val = p_ij * go_reg[e];

                        atomicAdd(&grad_k[kj_offset + d], gk_val);
                        atomicAdd(&grad_v[kj_offset + d], gv_val);
                    }
                }
            }
        }
    }

    if (i < N) {
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                grad_q[qi_offset + d] = from_float<scalar_t>(grad_q_acc[e]);
            }
        }
    }
}

}}

torch::Tensor routing_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& route_weights,
    int top_k
) {
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(route_weights);

    TORCH_CHECK(q.dim() == 4, "q must be 4D [B, H, N, D]");
    TORCH_CHECK(k.sizes() == q.sizes(), "k shape mismatch");
    TORCH_CHECK(v.sizes() == q.sizes(), "v shape mismatch");

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    TORCH_CHECK(D <= 128, "routing_attention_fwd: D=", D, " > 128 not supported");

    auto out = torch::empty_like(q);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto rw_c = route_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "routing_attn_fwd", ([&] {
            neuralops::cuda::routing_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                rw_c.data_ptr<scalar_t>(),
                B, H, N, D, top_k
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> routing_attn_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& route_weights,
    int top_k
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(route_weights);

    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    auto grad_q = torch::empty_like(q);
    auto grad_k = torch::zeros({B, H, N, D}, q.options().dtype(torch::kFloat32));
    auto grad_v = torch::zeros({B, H, N, D}, q.options().dtype(torch::kFloat32));

    if (B == 0 || H == 0 || N == 0 || D == 0) {
        return {grad_q, grad_k.to(k.dtype()), grad_v.to(v.dtype())};
    }

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto go_c = grad_out.contiguous();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();
    auto rw_c = route_weights.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "routing_attn_bwd", ([&] {
            neuralops::cuda::routing_attn_bwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<float>(),
                grad_v.data_ptr<float>(),
                go_c.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                rw_c.data_ptr<scalar_t>(),
                B, H, N, D, top_k
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_q, grad_k.to(k.dtype()), grad_v.to(v.dtype())};
}
