#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include "_common.cuh"

namespace neuralops { namespace cuda {

// ============================================================================
// Differential Attention Forward Kernel
// Form: A_1 = \text{Softmax}(Q_1 K_1^T / \sqrt{D}), \quad A_2 = \text{Softmax}(Q_2 K_2^T / \sqrt{D})
//       out_i = (A_1 - \lambda A_2) V
// ============================================================================
template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) diff_attn_fwd_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ k1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ k2,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float lambda_val
) {
    const int bh = blockIdx.y;
    if (bh >= B * H) return;

    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x % 32;
    const int i = blockIdx.x * 4 + warp_id;

    const int base_offset = bh * (N * D);
    const int qi_offset = base_offset + i * D;
    const float scale = rsqrtf(static_cast<float>(D));

    // Register cache for query vectors (up to D=128)
    constexpr int MAX_E = 4;
    float q1_reg[MAX_E];
    float q2_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (i < N && d < D) {
            q1_reg[e] = to_float(q1[qi_offset + d]) * scale;
            q2_reg[e] = to_float(q2[qi_offset + d]) * scale;
        } else {
            q1_reg[e] = 0.0f;
            q2_reg[e] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8; // +8 elements eliminates bank conflicts & ensures 16-byte alignment
    __shared__ float s_k1[TILE_K][D_PAD];
    __shared__ float s_k2[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    // Single-pass FlashAttention: fused online softmax rescaling and value accumulation
    float m1 = -1e30f, d1 = 0.0f;
    float m2 = -1e30f, d2 = 0.0f;
    float acc1[MAX_E] = {0.0f};
    float acc2[MAX_E] = {0.0f};

    const int tid = threadIdx.x; // 0..127
    const int max_block_i = blockIdx.x * 4 + 3;

    for (int tile_start = 0; tile_start < N && tile_start <= max_block_i; tile_start += TILE_K) {
        __syncthreads();

        // 128-bit aligned vector DRAM loads
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += 128) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k1 = reinterpret_cast<const uint4*>(k1 + g_off)[0];
                    const uint4 raw_k2 = reinterpret_cast<const uint4*>(k2 + g_off)[0];
                    const uint4 raw_v  = reinterpret_cast<const uint4*>(v  + g_off)[0];

                    const scalar_t* p_k1 = reinterpret_cast<const scalar_t*>(&raw_k1);
                    const scalar_t* p_k2 = reinterpret_cast<const scalar_t*>(&raw_k2);
                    const scalar_t* p_v  = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = to_float(p_k1[elem]);
                        s_k2[t][d_base + elem] = to_float(p_k2[elem]);
                        s_v[t][d_base + elem]  = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = 0.0f;
                        s_k2[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem]  = 0.0f;
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
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const float4 raw_k1 = reinterpret_cast<const float4*>(k1 + g_off)[0];
                    const float4 raw_k2 = reinterpret_cast<const float4*>(k2 + g_off)[0];
                    const float4 raw_v  = reinterpret_cast<const float4*>(v  + g_off)[0];

                    s_k1[t][d_base + 0] = raw_k1.x;
                    s_k1[t][d_base + 1] = raw_k1.y;
                    s_k1[t][d_base + 2] = raw_k1.z;
                    s_k1[t][d_base + 3] = raw_k1.w;

                    s_k2[t][d_base + 0] = raw_k2.x;
                    s_k2[t][d_base + 1] = raw_k2.y;
                    s_k2[t][d_base + 2] = raw_k2.z;
                    s_k2[t][d_base + 3] = raw_k2.w;

                    s_v[t][d_base + 0]  = raw_v.x;
                    s_v[t][d_base + 1]  = raw_v.y;
                    s_v[t][d_base + 2]  = raw_v.z;
                    s_v[t][d_base + 3]  = raw_v.w;
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 4; ++elem) {
                        s_k1[t][d_base + elem] = 0.0f;
                        s_k2[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem]  = 0.0f;
                    }
                }
            }
        } else {
            const int tile_elements = TILE_K * D;
            for (int idx = tid; idx < tile_elements; idx += 128) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k1[t][d] = to_float(k1[g_off]);
                    s_k2[t][d] = to_float(k2[g_off]);
                    s_v[t][d]  = to_float(v[g_off]);
                } else {
                    s_k1[t][d] = 0.0f;
                    s_k2[t][d] = 0.0f;
                    s_v[t][d]  = 0.0f;
                }
            }
        }
        __syncthreads();

        // Warp computes against SMEM tile
        if (i < N && tile_start <= i) {
            const int max_t = min(TILE_K - 1, i - tile_start);
            for (int t = 0; t <= max_t; ++t) {
                float dot1_local = 0.0f;
                float dot2_local = 0.0f;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot1_local = fmaf(q1_reg[e], s_k1[t][d], dot1_local);
                        dot2_local = fmaf(q2_reg[e], s_k2[t][d], dot2_local);
                    }
                }

                // Warp reduction across 32 lanes
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot1_local += __shfl_xor_sync(0xffffffff, dot1_local, offset);
                    dot2_local += __shfl_xor_sync(0xffffffff, dot2_local, offset);
                }

                // Online softmax 1 update
                const float m1_new = fmaxf(m1, dot1_local);
                const float alpha1 = expf(m1 - m1_new);
                const float p1 = expf(dot1_local - m1_new);
                d1 = d1 * alpha1 + p1;
                m1 = m1_new;

                // Online softmax 2 update
                const float m2_new = fmaxf(m2, dot2_local);
                const float alpha2 = expf(m2 - m2_new);
                const float p2 = expf(dot2_local - m2_new);
                d2 = d2 * alpha2 + p2;
                m2 = m2_new;

                // Scale prior accumulations by alpha and add current p * v
                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        const float v_val = s_v[t][d];
                        acc1[e] = acc1[e] * alpha1 + p1 * v_val;
                        acc2[e] = acc2[e] * alpha2 + p2 * v_val;
                    }
                }
            }
        }
    }

    if (i < N) {
        const float inv_d1 = (d1 > 1e-6f) ? (1.0f / d1) : 0.0f;
        const float inv_d2 = (d2 > 1e-6f) ? (1.0f / d2) : 0.0f;

        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float res = (acc1[e] * inv_d1) - lambda_val * (acc2[e] * inv_d2);
                out[qi_offset + d] = from_float<scalar_t>(res);
            }
        }
    }
}

// ============================================================================
// Differential Attention Backward Kernel (Analytical Adjoint with SMEM Tiling)
// ============================================================================
template <typename scalar_t>
__global__ void __launch_bounds__(128, 2) diff_attn_bwd_kernel(
    scalar_t* __restrict__ grad_q1,
    float* __restrict__ grad_k1,
    scalar_t* __restrict__ grad_q2,
    float* __restrict__ grad_k2,
    float* __restrict__ grad_v,
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ q1,
    const scalar_t* __restrict__ k1,
    const scalar_t* __restrict__ q2,
    const scalar_t* __restrict__ k2,
    const scalar_t* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const float lambda_val
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
    float q1_reg[MAX_E];
    float q2_reg[MAX_E];
    float go_reg[MAX_E];

    #pragma unroll
    for (int e = 0; e < MAX_E; ++e) {
        const int d = lane + e * 32;
        if (i < N && d < D) {
            q1_reg[e] = to_float(q1[qi_offset + d]) * scale;
            q2_reg[e] = to_float(q2[qi_offset + d]) * scale;
            go_reg[e] = to_float(grad_out[qi_offset + d]);
        } else {
            q1_reg[e] = 0.0f;
            q2_reg[e] = 0.0f;
            go_reg[e] = 0.0f;
        }
    }

    constexpr int TILE_K = 16;
    constexpr int D_PAD = 128 + 8;
    __shared__ float s_k1[TILE_K][D_PAD];
    __shared__ float s_k2[TILE_K][D_PAD];
    __shared__ float s_v[TILE_K][D_PAD];

    // Pass 1: Forward online softmax tracking
    float m1 = -1e30f, d1 = 0.0f;
    float m2 = -1e30f, d2 = 0.0f;
    float acc1[MAX_E] = {0.0f};
    float acc2[MAX_E] = {0.0f};

    const int tid = threadIdx.x;
    const int max_block_i = blockIdx.x * 4 + 3;

    for (int tile_start = 0; tile_start < N && tile_start <= max_block_i; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += 128) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k1 = reinterpret_cast<const uint4*>(k1 + g_off)[0];
                    const uint4 raw_k2 = reinterpret_cast<const uint4*>(k2 + g_off)[0];
                    const uint4 raw_v  = reinterpret_cast<const uint4*>(v  + g_off)[0];

                    const scalar_t* p_k1 = reinterpret_cast<const scalar_t*>(&raw_k1);
                    const scalar_t* p_k2 = reinterpret_cast<const scalar_t*>(&raw_k2);
                    const scalar_t* p_v  = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = to_float(p_k1[elem]);
                        s_k2[t][d_base + elem] = to_float(p_k2[elem]);
                        s_v[t][d_base + elem]  = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = 0.0f;
                        s_k2[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem]  = 0.0f;
                    }
                }
            }
        } else {
            const int tile_elements = TILE_K * D;
            for (int idx = tid; idx < tile_elements; idx += 128) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k1[t][d] = to_float(k1[g_off]);
                    s_k2[t][d] = to_float(k2[g_off]);
                    s_v[t][d]  = to_float(v[g_off]);
                } else {
                    s_k1[t][d] = 0.0f;
                    s_k2[t][d] = 0.0f;
                    s_v[t][d]  = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N && tile_start <= i) {
            const int max_t = min(TILE_K - 1, i - tile_start);
            for (int t = 0; t <= max_t; ++t) {
                float dot1_local = 0.0f;
                float dot2_local = 0.0f;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot1_local = fmaf(q1_reg[e], s_k1[t][d], dot1_local);
                        dot2_local = fmaf(q2_reg[e], s_k2[t][d], dot2_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot1_local += __shfl_xor_sync(0xffffffff, dot1_local, offset);
                    dot2_local += __shfl_xor_sync(0xffffffff, dot2_local, offset);
                }

                const float m1_new = fmaxf(m1, dot1_local);
                const float alpha1 = expf(m1 - m1_new);
                const float p1 = expf(dot1_local - m1_new);
                d1 = d1 * alpha1 + p1;
                m1 = m1_new;

                const float m2_new = fmaxf(m2, dot2_local);
                const float alpha2 = expf(m2 - m2_new);
                const float p2 = expf(dot2_local - m2_new);
                d2 = d2 * alpha2 + p2;
                m2 = m2_new;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        const float v_val = s_v[t][d];
                        acc1[e] = acc1[e] * alpha1 + p1 * v_val;
                        acc2[e] = acc2[e] * alpha2 + p2 * v_val;
                    }
                }
            }
        }
    }

    // Scalar D1 = sum_d (grad_out[i, d] * Out1[i, d]), D2 = -lambda * sum_d (grad_out[i, d] * Out2[i, d])
    float D1_val = 0.0f;
    float D2_val = 0.0f;
    if (i < N) {
        const float inv_d1 = (d1 > 1e-6f) ? (1.0f / d1) : 0.0f;
        const float inv_d2 = (d2 > 1e-6f) ? (1.0f / d2) : 0.0f;
        #pragma unroll
        for (int e = 0; e < MAX_E; ++e) {
            const int d = lane + e * 32;
            if (d < D) {
                const float o1_d = acc1[e] * inv_d1;
                const float o2_d = acc2[e] * inv_d2;
                D1_val += go_reg[e] * o1_d;
                D2_val += (-lambda_val) * go_reg[e] * o2_d;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            D1_val += __shfl_xor_sync(0xffffffff, D1_val, offset);
            D2_val += __shfl_xor_sync(0xffffffff, D2_val, offset);
        }
    }

    // Pass 2: Softmax adjoints & gradient accumulation
    float grad_q1_acc[MAX_E] = {0.0f};
    float grad_q2_acc[MAX_E] = {0.0f};

    for (int tile_start = 0; tile_start < N && tile_start <= max_block_i; tile_start += TILE_K) {
        __syncthreads();
        if (sizeof(scalar_t) == 2 && (D % 8 == 0)) {
            const int uint4_per_tok = D / 8;
            const int total_uint4 = TILE_K * uint4_per_tok;
            for (int idx = tid; idx < total_uint4; idx += 128) {
                const int t = idx / uint4_per_tok;
                const int chunk = idx % uint4_per_tok;
                const int d_base = chunk * 8;
                const int global_j = tile_start + t;

                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d_base;
                    const uint4 raw_k1 = reinterpret_cast<const uint4*>(k1 + g_off)[0];
                    const uint4 raw_k2 = reinterpret_cast<const uint4*>(k2 + g_off)[0];
                    const uint4 raw_v  = reinterpret_cast<const uint4*>(v  + g_off)[0];

                    const scalar_t* p_k1 = reinterpret_cast<const scalar_t*>(&raw_k1);
                    const scalar_t* p_k2 = reinterpret_cast<const scalar_t*>(&raw_k2);
                    const scalar_t* p_v  = reinterpret_cast<const scalar_t*>(&raw_v);

                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = to_float(p_k1[elem]);
                        s_k2[t][d_base + elem] = to_float(p_k2[elem]);
                        s_v[t][d_base + elem]  = to_float(p_v[elem]);
                    }
                } else {
                    #pragma unroll
                    for (int elem = 0; elem < 8; ++elem) {
                        s_k1[t][d_base + elem] = 0.0f;
                        s_k2[t][d_base + elem] = 0.0f;
                        s_v[t][d_base + elem]  = 0.0f;
                    }
                }
            }
        } else {
            const int tile_elements = TILE_K * D;
            for (int idx = tid; idx < tile_elements; idx += 128) {
                const int t = idx / D;
                const int d = idx % D;
                const int global_j = tile_start + t;
                if (global_j < N) {
                    const int g_off = base_offset + global_j * D + d;
                    s_k1[t][d] = to_float(k1[g_off]);
                    s_k2[t][d] = to_float(k2[g_off]);
                    s_v[t][d]  = to_float(v[g_off]);
                } else {
                    s_k1[t][d] = 0.0f;
                    s_k2[t][d] = 0.0f;
                    s_v[t][d]  = 0.0f;
                }
            }
        }
        __syncthreads();

        if (i < N && tile_start <= i) {
            const int max_t = min(TILE_K - 1, i - tile_start);
            const float inv_d1 = (d1 > 1e-6f) ? (1.0f / d1) : 0.0f;
            const float inv_d2 = (d2 > 1e-6f) ? (1.0f / d2) : 0.0f;

            for (int t = 0; t <= max_t; ++t) {
                float dot1_local = 0.0f;
                float dot2_local = 0.0f;
                float dA_local = 0.0f; // grad_out[i] . v[j]

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        dot1_local = fmaf(q1_reg[e], s_k1[t][d], dot1_local);
                        dot2_local = fmaf(q2_reg[e], s_k2[t][d], dot2_local);
                        dA_local   = fmaf(go_reg[e], s_v[t][d], dA_local);
                    }
                }

                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    dot1_local += __shfl_xor_sync(0xffffffff, dot1_local, offset);
                    dot2_local += __shfl_xor_sync(0xffffffff, dot2_local, offset);
                    dA_local   += __shfl_xor_sync(0xffffffff, dA_local, offset);
                }

                const float A1_ij = expf(dot1_local - m1) * inv_d1;
                const float A2_ij = expf(dot2_local - m2) * inv_d2;

                const float dA1 = dA_local;
                const float dA2 = (-lambda_val) * dA_local;

                const float dS1 = A1_ij * (dA1 - D1_val) * scale;
                const float dS2 = A2_ij * (dA2 - D2_val) * scale;

                const float diff_prob = A1_ij - lambda_val * A2_ij;

                const int global_j = tile_start + t;
                const int kj_offset = base_offset + global_j * D;

                #pragma unroll
                for (int e = 0; e < MAX_E; ++e) {
                    const int d = lane + e * 32;
                    if (d < D) {
                        grad_q1_acc[e] += dS1 * s_k1[t][d];
                        grad_q2_acc[e] += dS2 * s_k2[t][d];

                        const float gk1_val = dS1 * (to_float(q1[qi_offset + d]) * scale);
                        const float gk2_val = dS2 * (to_float(q2[qi_offset + d]) * scale);
                        const float gv_val  = diff_prob * go_reg[e];

                        atomicAdd(&grad_k1[kj_offset + d], gk1_val);
                        atomicAdd(&grad_k2[kj_offset + d], gk2_val);
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
                grad_q1[qi_offset + d] = from_float<scalar_t>(grad_q1_acc[e]);
                grad_q2[qi_offset + d] = from_float<scalar_t>(grad_q2_acc[e]);
            }
        }
    }
}

}} // namespace neuralops::cuda

torch::Tensor diff_attn_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
) {
    CHECK_INPUT(q1);
    CHECK_INPUT(k1);
    CHECK_INPUT(q2);
    CHECK_INPUT(k2);
    CHECK_INPUT(v);

    TORCH_CHECK(q1.dim() == 4, "q1 must be 4D [B, H, N, D]");
    TORCH_CHECK(k1.sizes() == q1.sizes(), "k1 shape mismatch");
    TORCH_CHECK(q2.sizes() == q1.sizes(), "q2 shape mismatch");
    TORCH_CHECK(k2.sizes() == q1.sizes(), "k2 shape mismatch");
    TORCH_CHECK(v.sizes() == q1.sizes(), "v shape mismatch");

    const int B = q1.size(0);
    const int H = q1.size(1);
    const int N = q1.size(2);
    const int D = q1.size(3);
    TORCH_CHECK(D <= 128, "differential_attention_fwd: D=", D, " exceeds max 128");

    auto out = torch::empty_like(q1);
    if (B == 0 || H == 0 || N == 0 || D == 0) return out;

    constexpr int THREADS = 128; // 4 warps per block -> 4 queries per block
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto q1_c = q1.contiguous();
    auto k1_c = k1.contiguous();
    auto q2_c = q2.contiguous();
    auto k2_c = k2.contiguous();
    auto v_c = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q1.scalar_type(), "diff_attn_fwd", ([&] {
            neuralops::cuda::diff_attn_fwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                out.data_ptr<scalar_t>(),
                q1_c.data_ptr<scalar_t>(),
                k1_c.data_ptr<scalar_t>(),
                q2_c.data_ptr<scalar_t>(),
                k2_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, static_cast<float>(lambda_val)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

std::vector<torch::Tensor> diff_attn_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
) {
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q1);
    CHECK_INPUT(k1);
    CHECK_INPUT(q2);
    CHECK_INPUT(k2);
    CHECK_INPUT(v);

    const int B = q1.size(0);
    const int H = q1.size(1);
    const int N = q1.size(2);
    const int D = q1.size(3);

    auto grad_q1 = torch::empty_like(q1);
    auto grad_q2 = torch::empty_like(q2);
    auto grad_k1_f = torch::zeros_like(q1, q1.options().dtype(torch::kFloat32));
    auto grad_k2_f = torch::zeros_like(q2, q2.options().dtype(torch::kFloat32));
    auto grad_v_f  = torch::zeros_like(v,  v.options().dtype(torch::kFloat32));

    constexpr int THREADS = 128;
    constexpr int WARPS_PER_BLOCK = THREADS / 32;
    const int blocks_x = (N + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    dim3 grid(blocks_x, B * H);

    auto stream = c10::cuda::getCurrentCUDAStream();
    auto go_c = grad_out.contiguous();
    auto q1_c = q1.contiguous();
    auto k1_c = k1.contiguous();
    auto q2_c = q2.contiguous();
    auto k2_c = k2.contiguous();
    auto v_c  = v.contiguous();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16,
        q1.scalar_type(), "diff_attn_bwd", ([&] {
            neuralops::cuda::diff_attn_bwd_kernel<scalar_t><<<grid, THREADS, 0, stream>>>(
                grad_q1.data_ptr<scalar_t>(),
                grad_k1_f.data_ptr<float>(),
                grad_q2.data_ptr<scalar_t>(),
                grad_k2_f.data_ptr<float>(),
                grad_v_f.data_ptr<float>(),
                go_c.data_ptr<scalar_t>(),
                q1_c.data_ptr<scalar_t>(),
                k1_c.data_ptr<scalar_t>(),
                q2_c.data_ptr<scalar_t>(),
                k2_c.data_ptr<scalar_t>(),
                v_c.data_ptr<scalar_t>(),
                B, H, N, D, static_cast<float>(lambda_val)
            );
        })
    );

    CUDA_KERNEL_LAUNCH_CHECK();

    auto grad_k1 = grad_k1_f.to(q1.dtype());
    auto grad_k2 = grad_k2_f.to(q2.dtype());
    auto grad_v  = grad_v_f.to(v.dtype());

    return {grad_q1, grad_k1, grad_q2, grad_k2, grad_v};
}
