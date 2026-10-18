#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>
#include "_common.cuh"

// ============================================================================
// Block Sparse Attention Forward Kernel
// SMEM block tiling (+8 padding), 128-bit vector loads, and block mask caching
// ============================================================================
template<int HEAD_DIM, int BLOCK_SIZE>
__launch_bounds__(128, 2)
__global__ void block_sparse_attn_warp_kernel(
    float* __restrict__ out,               // [B, H, N, HEAD_DIM]
    const float* __restrict__ q,           // [B, H, N, HEAD_DIM]
    const float* __restrict__ k,           // [B, H, N, HEAD_DIM]
    const float* __restrict__ v,           // [B, H, N, HEAD_DIM]
    const int32_t* __restrict__ block_mask,// [num_q_blocks, num_kv_blocks]
    const int B,
    const int H,
    const int N,
    const int num_kv_blocks,
    const float scale
) {
    constexpr int NUM_WARPS = 4;
    constexpr int QUERIES_PER_WARP = BLOCK_SIZE / NUM_WARPS;
    constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / 32;
    constexpr int STRIDE = HEAD_DIM + 8; // 16-byte aligned + 8-bank shift

    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid_in_block = warp_id * 32 + lane;
    const int total_threads = NUM_WARPS * 32;

    if (b >= B || h >= H) return;

    // Dynamic shared memory layout:
    // K_smem: [BLOCK_SIZE][STRIDE]
    // V_smem: [BLOCK_SIZE][STRIDE]
    // s_active_kv: [num_kv_blocks]
    extern __shared__ char smem_raw[];
    float* K_smem = reinterpret_cast<float*>(smem_raw);
    float* V_smem = reinterpret_cast<float*>(smem_raw + BLOCK_SIZE * STRIDE * sizeof(float));
    int32_t* s_active_kv = reinterpret_cast<int32_t*>(smem_raw + 2 * BLOCK_SIZE * STRIDE * sizeof(float));

    __shared__ int s_num_active;

    // Block sparse mask caching: cache active KV blocks for this query block into SMEM
    if (tid_in_block == 0) {
        int count = 0;
        const int mask_row_off = q_block_idx * num_kv_blocks;
        for (int kv_b = 0; kv_b < num_kv_blocks; ++kv_b) {
            if (block_mask[mask_row_off + kv_b] != 0) {
                s_active_kv[count++] = kv_b;
            }
        }
        s_num_active = count;
    }
    __syncthreads();

    const int num_act = s_num_active;
    const int kv_head_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM);

    // Local registers for queries assigned to this warp
    float q_local[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float acc[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float m_prev[QUERIES_PER_WARP];
    float l_prev[QUERIES_PER_WARP];
    int q_indices[QUERIES_PER_WARP];

    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        q_indices[qi] = q_block_idx * BLOCK_SIZE + warp_id * QUERIES_PER_WARP + qi;
        m_prev[qi] = -FLT_MAX;
        l_prev[qi] = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            acc[qi][e] = 0.0f;
            if (q_indices[qi] < N) {
                const int q_off = kv_head_base + q_indices[qi] * HEAD_DIM + lane + e * 32;
                q_local[qi][e] = q[q_off];
            } else {
                q_local[qi][e] = 0.0f;
            }
        }
    }

    constexpr int VEC4_PER_TOK = HEAD_DIM / 4;
    constexpr int TOTAL_VEC4 = BLOCK_SIZE * VEC4_PER_TOK;

    // Loop over cached active KV blocks
    for (int act_idx = 0; act_idx < num_act; ++act_idx) {
        const int kv_b = s_active_kv[act_idx];
        const int k_start = kv_b * BLOCK_SIZE;

        // Cooperatively load active K and V into SMEM with 128-bit aligned float4 vector loads
        __syncthreads();
        for (int idx = tid_in_block; idx < TOTAL_VEC4; idx += total_threads) {
            const int tok_in_block = idx / VEC4_PER_TOK;
            const int vec_in_tok = idx % VEC4_PER_TOK;
            const int dim_start = vec_in_tok * 4;
            const int global_tok = k_start + tok_in_block;

            if (global_tok < N) {
                const int g_off = kv_head_base + global_tok * HEAD_DIM + dim_start;
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&k[g_off]);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&v[g_off]);
            } else {
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        const int tokens_in_block = min(BLOCK_SIZE, N - k_start);

        // Process all queries in this warp against the loaded active KV block
        #pragma unroll
        for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
            const int cur_q_idx = q_indices[qi];
            if (cur_q_idx >= N) continue;

            for (int kt = 0; kt < tokens_in_block; ++kt) {
                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    dot += q_local[qi][e] * K_smem[kt * STRIDE + lane + e * 32];
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                const float qk = dot * scale;
                const float m_curr = fmaxf(m_prev[qi], qk);
                const float alpha = expf(m_prev[qi] - m_curr);
                const float p = expf(qk - m_curr);

                l_prev[qi] = l_prev[qi] * alpha + p;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    acc[qi][e] = acc[qi][e] * alpha + p * V_smem[kt * STRIDE + lane + e * 32];
                }
                m_prev[qi] = m_curr;
            }
        }
    }

    // Write final normalized outputs to global memory
    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_indices[qi];
        if (cur_q_idx < N) {
            const int out_base = kv_head_base + cur_q_idx * HEAD_DIM;
            if (l_prev[qi] > 0.0f) {
                const float inv_l = 1.0f / l_prev[qi];
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    out[out_base + lane + e * 32] = acc[qi][e] * inv_l;
                }
            } else {
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    out[out_base + lane + e * 32] = 0.0f;
                }
            }
        }
    }
}

// ============================================================================
// Block Sparse Attention Backward Kernel (Analytical Adjoint with SMEM Tiling)
// Computes exact analytical gradients for Q, K, V with 2-pass online softmax adjoints
// ============================================================================
template<int HEAD_DIM, int BLOCK_SIZE>
__launch_bounds__(128, 2)
__global__ void sparse_attention_bwd_kernel(
    float* __restrict__ grad_q,            // [B, H, N, HEAD_DIM]
    float* __restrict__ grad_k,            // [B, H, N, HEAD_DIM]
    float* __restrict__ grad_v,            // [B, H, N, HEAD_DIM]
    const float* __restrict__ grad_out,    // [B, H, N, HEAD_DIM]
    const float* __restrict__ q,           // [B, H, N, HEAD_DIM]
    const float* __restrict__ k,           // [B, H, N, HEAD_DIM]
    const float* __restrict__ v,           // [B, H, N, HEAD_DIM]
    const int32_t* __restrict__ block_mask,// [num_q_blocks, num_kv_blocks]
    const int B,
    const int H,
    const int N,
    const int num_kv_blocks,
    const float scale
) {
    constexpr int NUM_WARPS = 4;
    constexpr int QUERIES_PER_WARP = BLOCK_SIZE / NUM_WARPS;
    constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / 32;
    constexpr int STRIDE = HEAD_DIM + 8; // 16-byte aligned + 8-bank shift

    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int tid_in_block = warp_id * 32 + lane;
    const int total_threads = NUM_WARPS * 32;

    if (b >= B || h >= H) return;

    extern __shared__ char smem_raw[];
    float* K_smem = reinterpret_cast<float*>(smem_raw);
    float* V_smem = reinterpret_cast<float*>(smem_raw + BLOCK_SIZE * STRIDE * sizeof(float));
    int32_t* s_active_kv = reinterpret_cast<int32_t*>(smem_raw + 2 * BLOCK_SIZE * STRIDE * sizeof(float));

    __shared__ int s_num_active;

    // Block sparse mask caching: cache active KV blocks for this query block into SMEM
    if (tid_in_block == 0) {
        int count = 0;
        const int mask_row_off = q_block_idx * num_kv_blocks;
        for (int kv_b = 0; kv_b < num_kv_blocks; ++kv_b) {
            if (block_mask[mask_row_off + kv_b] != 0) {
                s_active_kv[count++] = kv_b;
            }
        }
        s_num_active = count;
    }
    __syncthreads();

    const int num_act = s_num_active;
    const int kv_head_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM);

    // Query indices for this warp
    int q_indices[QUERIES_PER_WARP];
    float q_local[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float go_local[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float acc[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float m_prev[QUERIES_PER_WARP];
    float l_prev[QUERIES_PER_WARP];

    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        q_indices[qi] = q_block_idx * BLOCK_SIZE + warp_id * QUERIES_PER_WARP + qi;
        m_prev[qi] = -FLT_MAX;
        l_prev[qi] = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            acc[qi][e] = 0.0f;
            if (q_indices[qi] < N) {
                const int q_off = kv_head_base + q_indices[qi] * HEAD_DIM + lane + e * 32;
                q_local[qi][e] = q[q_off];
                go_local[qi][e] = grad_out[q_off];
            } else {
                q_local[qi][e] = 0.0f;
                go_local[qi][e] = 0.0f;
            }
        }
    }

    if (num_act == 0) {
        // No active blocks for this query block; grad_q is 0
        #pragma unroll
        for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
            const int cur_q_idx = q_indices[qi];
            if (cur_q_idx < N) {
                const int q_off = kv_head_base + cur_q_idx * HEAD_DIM;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    grad_q[q_off + lane + e * 32] = 0.0f;
                }
            }
        }
        return;
    }

    constexpr int VEC4_PER_TOK = HEAD_DIM / 4;
    constexpr int TOTAL_VEC4 = BLOCK_SIZE * VEC4_PER_TOK;

    // ========================================================================
    // Pass 1: Forward online softmax tracking to compute m, l, and D_i
    // ========================================================================
    for (int act_idx = 0; act_idx < num_act; ++act_idx) {
        const int kv_b = s_active_kv[act_idx];
        const int k_start = kv_b * BLOCK_SIZE;

        __syncthreads();
        for (int idx = tid_in_block; idx < TOTAL_VEC4; idx += total_threads) {
            const int tok_in_block = idx / VEC4_PER_TOK;
            const int vec_in_tok = idx % VEC4_PER_TOK;
            const int dim_start = vec_in_tok * 4;
            const int global_tok = k_start + tok_in_block;

            if (global_tok < N) {
                const int g_off = kv_head_base + global_tok * HEAD_DIM + dim_start;
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&k[g_off]);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&v[g_off]);
            } else {
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        const int tokens_in_block = min(BLOCK_SIZE, N - k_start);

        #pragma unroll
        for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
            const int cur_q_idx = q_indices[qi];
            if (cur_q_idx >= N) continue;

            for (int kt = 0; kt < tokens_in_block; ++kt) {
                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    dot += q_local[qi][e] * K_smem[kt * STRIDE + lane + e * 32];
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                const float qk = dot * scale;
                const float m_curr = fmaxf(m_prev[qi], qk);
                const float alpha = expf(m_prev[qi] - m_curr);
                const float p = expf(qk - m_curr);

                l_prev[qi] = l_prev[qi] * alpha + p;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    acc[qi][e] = acc[qi][e] * alpha + p * V_smem[kt * STRIDE + lane + e * 32];
                }
                m_prev[qi] = m_curr;
            }
        }
    }

    // Scalar D_i = sum_d (grad_out[i, d] * Out[i, d]) for each query
    float D_val[QUERIES_PER_WARP];
    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_indices[qi];
        D_val[qi] = 0.0f;
        if (cur_q_idx < N && l_prev[qi] > 0.0f) {
            const float inv_l = 1.0f / l_prev[qi];
            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                const float out_d = acc[qi][e] * inv_l;
                D_val[qi] += go_local[qi][e] * out_d;
            }
            #pragma unroll
            for (int mask = 16; mask > 0; mask /= 2) {
                D_val[qi] += __shfl_xor_sync(0xffffffff, D_val[qi], mask);
            }
        }
    }

    // ========================================================================
    // Pass 2: Softmax adjoint & gradient accumulation for Q, K, V
    // ========================================================================
    float grad_q_acc[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            grad_q_acc[qi][e] = 0.0f;
        }
    }

    for (int act_idx = 0; act_idx < num_act; ++act_idx) {
        const int kv_b = s_active_kv[act_idx];
        const int k_start = kv_b * BLOCK_SIZE;

        __syncthreads();
        for (int idx = tid_in_block; idx < TOTAL_VEC4; idx += total_threads) {
            const int tok_in_block = idx / VEC4_PER_TOK;
            const int vec_in_tok = idx % VEC4_PER_TOK;
            const int dim_start = vec_in_tok * 4;
            const int global_tok = k_start + tok_in_block;

            if (global_tok < N) {
                const int g_off = kv_head_base + global_tok * HEAD_DIM + dim_start;
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&k[g_off]);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    *reinterpret_cast<const float4*>(&v[g_off]);
            } else {
                *reinterpret_cast<float4*>(&K_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                *reinterpret_cast<float4*>(&V_smem[tok_in_block * STRIDE + dim_start]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        const int tokens_in_block = min(BLOCK_SIZE, N - k_start);

        #pragma unroll
        for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
            const int cur_q_idx = q_indices[qi];
            if (cur_q_idx >= N || l_prev[qi] <= 0.0f) continue;

            const float inv_l = 1.0f / l_prev[qi];
            const float m_i = m_prev[qi];
            const float D_i = D_val[qi];

            for (int kt = 0; kt < tokens_in_block; ++kt) {
                float dot = 0.0f;
                float dp = 0.0f;

                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    const int d = lane + e * 32;
                    dot += q_local[qi][e] * K_smem[kt * STRIDE + d];
                    dp  += go_local[qi][e] * V_smem[kt * STRIDE + d];
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                    dp  += __shfl_xor_sync(0xffffffff, dp, mask);
                }

                const float qk = dot * scale;
                const float p = expf(qk - m_i) * inv_l;
                const float dS = p * (dp - D_i) * scale;

                const int global_j = k_start + kt;
                const int kv_tok_base = kv_head_base + global_j * HEAD_DIM;

                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    const int d = lane + e * 32;
                    grad_q_acc[qi][e] += dS * K_smem[kt * STRIDE + d];

                    const float gk_val = dS * q_local[qi][e];
                    const float gv_val = p * go_local[qi][e];

                    atomicAdd(&grad_k[kv_tok_base + d], gk_val);
                    atomicAdd(&grad_v[kv_tok_base + d], gv_val);
                }
            }
        }
    }

    // Write final accumulated grad_q directly to DRAM (coalesced)
    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_indices[qi];
        if (cur_q_idx < N) {
            const int q_off = kv_head_base + cur_q_idx * HEAD_DIM;
            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                grad_q[q_off + lane + e * 32] = grad_q_acc[qi][e];
            }
        }
    }
}

// ============================================================================
// Forward CUDA Host Dispatcher
// ============================================================================
void block_sparse_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
) {
    CHECK_INPUT(out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(block_mask);

    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int N = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int num_q_blocks = (N + block_size - 1) / block_size;
    const int num_kv_blocks = static_cast<int>(block_mask.size(1));

    if (B <= 0 || H <= 0 || N <= 0 || D <= 0 || num_q_blocks <= 0 || num_kv_blocks <= 0) {
        return;
    }

    dim3 grid(num_q_blocks, H, B);
    dim3 block(32, 4); // 4 warps = 128 threads
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_SPARSE_KERNEL(HEAD_DIM_VAL, BLOCK_SIZE_VAL) \
        do { \
            const size_t active_mask_bytes = num_kv_blocks * sizeof(int32_t); \
            const size_t smem_bytes = 2 * (BLOCK_SIZE_VAL) * ((HEAD_DIM_VAL) + 8) * sizeof(float) + active_mask_bytes; \
            auto kernel_func = block_sparse_attn_warp_kernel<HEAD_DIM_VAL, BLOCK_SIZE_VAL>; \
            if (smem_bytes > 48 * 1024) { \
                cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_func), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)); \
            } \
            kernel_func<<<grid, block, smem_bytes, stream>>>( \
                out.data_ptr<float>(), \
                q.data_ptr<float>(), \
                k.data_ptr<float>(), \
                v.data_ptr<float>(), \
                block_mask.data_ptr<int32_t>(), \
                B, H, N, num_kv_blocks, scale \
            ); \
        } while (0)

    if (block_size == 16) {
        if (D == 32) {
            LAUNCH_SPARSE_KERNEL(32, 16);
        } else if (D == 64) {
            LAUNCH_SPARSE_KERNEL(64, 16);
        } else if (D == 128) {
            LAUNCH_SPARSE_KERNEL(128, 16);
        }
    } else if (block_size == 32) {
        if (D == 32) {
            LAUNCH_SPARSE_KERNEL(32, 32);
        } else if (D == 64) {
            LAUNCH_SPARSE_KERNEL(64, 32);
        } else if (D == 128) {
            LAUNCH_SPARSE_KERNEL(128, 32);
        }
    } else if (block_size == 64) {
        if (D == 32) {
            LAUNCH_SPARSE_KERNEL(32, 64);
        } else if (D == 64) {
            LAUNCH_SPARSE_KERNEL(64, 64);
        } else if (D == 128) {
            LAUNCH_SPARSE_KERNEL(128, 64);
        }
    }

    #undef LAUNCH_SPARSE_KERNEL

    CUDA_KERNEL_LAUNCH_CHECK();
}

// ============================================================================
// Backward CUDA Host Dispatcher
// ============================================================================
void block_sparse_attn_bwd_cuda(
    torch::Tensor& grad_q,
    torch::Tensor& grad_k,
    torch::Tensor& grad_v,
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
) {
    CHECK_INPUT(grad_q);
    CHECK_INPUT(grad_k);
    CHECK_INPUT(grad_v);
    CHECK_INPUT(grad_out);
    CHECK_INPUT(q);
    CHECK_INPUT(k);
    CHECK_INPUT(v);
    CHECK_INPUT(block_mask);

    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int N = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int num_q_blocks = (N + block_size - 1) / block_size;
    const int num_kv_blocks = static_cast<int>(block_mask.size(1));

    if (B <= 0 || H <= 0 || N <= 0 || D <= 0 || num_q_blocks <= 0 || num_kv_blocks <= 0) {
        return;
    }

    dim3 grid(num_q_blocks, H, B);
    dim3 block(32, 4); // 4 warps = 128 threads
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    #define LAUNCH_SPARSE_BWD_KERNEL(HEAD_DIM_VAL, BLOCK_SIZE_VAL) \
        do { \
            const size_t active_mask_bytes = num_kv_blocks * sizeof(int32_t); \
            const size_t smem_bytes = 2 * (BLOCK_SIZE_VAL) * ((HEAD_DIM_VAL) + 8) * sizeof(float) + active_mask_bytes; \
            auto kernel_func = sparse_attention_bwd_kernel<HEAD_DIM_VAL, BLOCK_SIZE_VAL>; \
            if (smem_bytes > 48 * 1024) { \
                cudaFuncSetAttribute(reinterpret_cast<const void*>(kernel_func), cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_bytes)); \
            } \
            kernel_func<<<grid, block, smem_bytes, stream>>>( \
                grad_q.data_ptr<float>(), \
                grad_k.data_ptr<float>(), \
                grad_v.data_ptr<float>(), \
                grad_out.data_ptr<float>(), \
                q.data_ptr<float>(), \
                k.data_ptr<float>(), \
                v.data_ptr<float>(), \
                block_mask.data_ptr<int32_t>(), \
                B, H, N, num_kv_blocks, scale \
            ); \
        } while (0)

    if (block_size == 16) {
        if (D == 32) {
            LAUNCH_SPARSE_BWD_KERNEL(32, 16);
        } else if (D == 64) {
            LAUNCH_SPARSE_BWD_KERNEL(64, 16);
        } else if (D == 128) {
            LAUNCH_SPARSE_BWD_KERNEL(128, 16);
        }
    } else if (block_size == 32) {
        if (D == 32) {
            LAUNCH_SPARSE_BWD_KERNEL(32, 32);
        } else if (D == 64) {
            LAUNCH_SPARSE_BWD_KERNEL(64, 32);
        } else if (D == 128) {
            LAUNCH_SPARSE_BWD_KERNEL(128, 32);
        }
    } else if (block_size == 64) {
        if (D == 32) {
            LAUNCH_SPARSE_BWD_KERNEL(32, 64);
        } else if (D == 64) {
            LAUNCH_SPARSE_BWD_KERNEL(64, 64);
        } else if (D == 128) {
            LAUNCH_SPARSE_BWD_KERNEL(128, 64);
        }
    }

    #undef LAUNCH_SPARSE_BWD_KERNEL

    CUDA_KERNEL_LAUNCH_CHECK();
}
