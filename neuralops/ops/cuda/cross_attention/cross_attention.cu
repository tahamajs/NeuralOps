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
#include <vector>
#include "_common.cuh"

namespace neuralops {
namespace cuda {

template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) cross_attention_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int B,
    const int H,
    const int L_q,
    const int L_kv,
    const int D,
    const float scale
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int q_offset = bh * (L_q * D) + i * D;
    const int kv_head_offset = bh * (L_kv * D);

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];
    #pragma unroll
    for (int k = 0; k < MAX_E; ++k) {
        const int d = lane_id + k * 32;
        q_reg[k] = (i < L_q && d < D) ? to_float(Q[q_offset + d]) * scale : 0.0f;
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int tid = threadIdx.x;
    const int total_threads = 128;
    const int tile_elements = TILE_K * D;

    for (int tile_start = 0; tile_start < L_kv; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(K + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(V + g_off)[0];

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
                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d;
                    s_k[t][d] = to_float(K[g_off]);
                    s_v[t][d] = to_float(V[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < L_q) {
            const int max_t = min(TILE_K, L_kv - tile_start);
            for (int t = 0; t < max_t; ++t) {
                float local_dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        local_dot = fmaf(q_reg[k], s_k[t][d], local_dot);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_dot += __shfl_xor_sync(0xffffffff, local_dot, offset);
                }

                const float m_new = fmaxf(max_s, local_dot);
                const float alpha = expf(max_s - m_new);
                const float p = expf(local_dot - m_new);
                sum_exp = sum_exp * alpha + p;
                max_s = m_new;

                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        acc[k] = acc[k] * alpha + p * s_v[t][d];
                    }
                }
            }
        }
    }

    if (i < L_q) {
        const float inv_sum = (sum_exp > 1e-6f) ? (1.0f / sum_exp) : 0.0f;
        const int out_offset = bh * (L_q * D) + i * D;
        #pragma unroll
        for (int k = 0; k < MAX_E; ++k) {
            const int d = lane_id + k * 32;
            if (d < D) {
                out[out_offset + d] = from_float<scalar_t>(acc[k] * inv_sum);
            }
        }
    }
}

template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) cross_attention_bwd_kernel(
    scalar_t* __restrict__ grad_q,
    float* __restrict__ grad_k,
    float* __restrict__ grad_v,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ Q,
    const scalar_t* __restrict__ K,
    const scalar_t* __restrict__ V,
    const int B,
    const int H,
    const int L_q,
    const int L_kv,
    const int D,
    const float scale
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane_id = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int q_offset = bh * (L_q * D) + i * D;
    const int kv_head_offset = bh * (L_kv * D);

    constexpr int MAX_E = 4;
    float q_reg[MAX_E];
    float go_reg[MAX_E];

    #pragma unroll
    for (int k = 0; k < MAX_E; ++k) {
        const int d = lane_id + k * 32;
        if (i < L_q && d < D) {
            q_reg[k] = to_float(Q[q_offset + d]) * scale;
            go_reg[k] = to_float(grad_out[q_offset + d]);
        } else {
            q_reg[k] = 0.0f;
            go_reg[k] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    float max_s = -1e38f;
    float sum_exp = 0.0f;
    float acc[MAX_E] = {0.0f};

    const int tid = threadIdx.x;
    const int total_threads = 128;
    const int tile_elements = TILE_K * D;

    for (int tile_start = 0; tile_start < L_kv; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(K + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(V + g_off)[0];

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
                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d;
                    s_k[t][d] = to_float(K[g_off]);
                    s_v[t][d] = to_float(V[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < L_q) {
            const int max_t = min(TILE_K, L_kv - tile_start);
            for (int t = 0; t < max_t; ++t) {
                float local_dot = 0.0f;
                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        local_dot = fmaf(q_reg[k], s_k[t][d], local_dot);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_dot += __shfl_xor_sync(0xffffffff, local_dot, offset);
                }

                const float m_new = fmaxf(max_s, local_dot);
                const float alpha = expf(max_s - m_new);
                const float p = expf(local_dot - m_new);
                sum_exp = sum_exp * alpha + p;
                max_s = m_new;

                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        acc[k] = acc[k] * alpha + p * s_v[t][d];
                    }
                }
            }
        }
    }

    float D_val = 0.0f;
    if (i < L_q && sum_exp > 1e-6f) {
        const float inv_sum = 1.0f / sum_exp;
        #pragma unroll
        for (int k = 0; k < MAX_E; ++k) {
            const int d = lane_id + k * 32;
            if (d < D) {
                const float o_d = acc[k] * inv_sum;
                D_val += go_reg[k] * o_d;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            D_val += __shfl_xor_sync(0xffffffff, D_val, offset);
        }
    }

    float grad_q_acc[MAX_E] = {0.0f};

    for (int tile_start = 0; tile_start < L_kv; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += total_threads) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d_base;
                    const uint4 raw_k = reinterpret_cast<const uint4*>(K + g_off)[0];
                    const uint4 raw_v = reinterpret_cast<const uint4*>(V + g_off)[0];

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
                if (global_j < L_kv) {
                    const int g_off = kv_head_offset + global_j * D + d;
                    s_k[t][d] = to_float(K[g_off]);
                    s_v[t][d] = to_float(V[g_off]);
                } else {
                    s_k[t][d] = 0.0f;
                    s_v[t][d] = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < L_q && sum_exp > 1e-6f) {
            const int max_t = min(TILE_K, L_kv - tile_start);
            const float inv_sum = 1.0f / sum_exp;

            for (int t = 0; t < max_t; ++t) {
                float local_dot = 0.0f;
                float dp_local = 0.0f;

                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        local_dot = fmaf(q_reg[k], s_k[t][d], local_dot);
                        dp_local  = fmaf(go_reg[k], s_v[t][d], dp_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_dot += __shfl_xor_sync(0xffffffff, local_dot, offset);
                    dp_local  += __shfl_xor_sync(0xffffffff, dp_local, offset);
                }

                const float p_ij = expf(local_dot - max_s) * inv_sum;
                const float dS_ij = p_ij * (dp_local - D_val) * scale;

                const int global_j = tile_start + t;
                const int kj_offset = kv_head_offset + global_j * D;

                #pragma unroll
                for (int k = 0; k < MAX_E; ++k) {
                    const int d = lane_id + k * 32;
                    if (d < D) {
                        grad_q_acc[k] += dS_ij * s_k[t][d];

                        const float gk_val = dS_ij * to_float(Q[q_offset + d]);
                        const float gv_val = p_ij * go_reg[k];

                        atomicAdd(&grad_k[kj_offset + d], gk_val);
                        atomicAdd(&grad_v[kj_offset + d], gv_val);
                    }
                }
            }
        }
    }

    if (i < L_q) {
        #pragma unroll
        for (int k = 0; k < MAX_E; ++k) {
            const int d = lane_id + k * 32;
            if (d < D) {
                grad_q[q_offset + d] = from_float<scalar_t>(grad_q_acc[k]);
            }
        }
    }
}

}
}

torch::Tensor cross_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L_q = q.size(2);
    const int D = q.size(3);
    const int L_kv = k.size(2);
    TORCH_CHECK(D <= 128, "cross_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q);
    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (L_q + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cross_attention_fwd", ([&] {
            neuralops::cuda::cross_attention_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, L_q, L_kv, D, scale
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> cross_attention_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
) {
    TORCH_CHECK(grad_out.is_cuda(), "grad_out must be a CUDA tensor");
    TORCH_CHECK(q.is_cuda(), "q must be a CUDA tensor");
    TORCH_CHECK(k.is_cuda(), "k must be a CUDA tensor");
    TORCH_CHECK(v.is_cuda(), "v must be a CUDA tensor");

    const int B = q.size(0);
    const int H = q.size(1);
    const int L_q = q.size(2);
    const int D = q.size(3);
    const int L_kv = k.size(2);

    auto grad_q = torch::empty_like(q);
    auto grad_k = torch::zeros({B, H, L_kv, D}, q.options().dtype(torch::kFloat32));
    auto grad_v = torch::zeros({B, H, L_kv, D}, q.options().dtype(torch::kFloat32));

    if (B == 0 || H == 0 || L_q == 0 || L_kv == 0 || D == 0) {
        return {grad_q, grad_k.to(k.dtype()), grad_v.to(v.dtype())};
    }

    const float scale = 1.0f / sqrtf(static_cast<float>(D));

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (L_q + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);
    auto stream = c10::cuda::getCurrentCUDAStream();

    auto go_c = grad_out.contiguous();
    auto q_c = q.contiguous();
    auto k_c = k.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q.scalar_type(), "cross_attention_bwd", ([&] {
            neuralops::cuda::cross_attention_bwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                grad_q.data_ptr<scalar_t>(),
                grad_k.data_ptr<float>(),
                grad_v.data_ptr<float>(),
                go_c.data_ptr<scalar_t>(),
                q_c.data_ptr<scalar_t>(),
                k_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, L_q, L_kv, D, scale
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return {grad_q, grad_k.to(k.dtype()), grad_v.to(v.dtype())};
}
