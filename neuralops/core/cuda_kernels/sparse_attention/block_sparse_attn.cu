#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

template<int HEAD_DIM, int BLOCK_SIZE>
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

    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    if (b >= B || h >= H) return;

    // Shared memory with +1 padding to eliminate bank conflicts
    __shared__ float K_smem[BLOCK_SIZE][HEAD_DIM + 1];
    __shared__ float V_smem[BLOCK_SIZE][HEAD_DIM + 1];

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
                const int q_off = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_indices[qi] * HEAD_DIM + lane + e * 32;
                q_local[qi][e] = q[q_off];
            } else {
                q_local[qi][e] = 0.0f;
            }
        }
    }

    const int tid_in_block = warp_id * 32 + lane;
    const int total_threads = NUM_WARPS * 32;
    const int elements_per_kv_block = BLOCK_SIZE * HEAD_DIM;
    const int kv_head_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM);

    for (int kv_b = 0; kv_b < num_kv_blocks; ++kv_b) {
        const int is_active = __ldg(&block_mask[q_block_idx * num_kv_blocks + kv_b]);
        if (!is_active) {
            continue;
        }

        const int k_start = kv_b * BLOCK_SIZE;

        // Cooperatively load active K and V block into SMEM
        __syncthreads();
        for (int idx = tid_in_block; idx < elements_per_kv_block; idx += total_threads) {
            const int tok_in_block = idx / HEAD_DIM;
            const int dim_in_tok = idx % HEAD_DIM;
            const int global_tok = k_start + tok_in_block;

            if (global_tok < N) {
                const int g_off = kv_head_base + global_tok * HEAD_DIM + dim_in_tok;
                K_smem[tok_in_block][dim_in_tok] = k[g_off];
                V_smem[tok_in_block][dim_in_tok] = v[g_off];
            } else {
                K_smem[tok_in_block][dim_in_tok] = 0.0f;
                V_smem[tok_in_block][dim_in_tok] = 0.0f;
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
                    dot += q_local[qi][e] * K_smem[kt][lane + e * 32];
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                float qk = dot * scale;
                float m_curr = fmaxf(m_prev[qi], qk);
                float alpha = expf(m_prev[qi] - m_curr);
                float p = expf(qk - m_curr);

                l_prev[qi] = l_prev[qi] * alpha + p;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    acc[qi][e] = acc[qi][e] * alpha + p * V_smem[kt][lane + e * 32];
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
            const int out_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + cur_q_idx * HEAD_DIM;
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

void block_sparse_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
) {
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
        block_sparse_attn_warp_kernel<HEAD_DIM_VAL, BLOCK_SIZE_VAL><<<grid, block, 0, stream>>>( \
            out.data_ptr<float>(), \
            q.data_ptr<float>(), \
            k.data_ptr<float>(), \
            v.data_ptr<float>(), \
            block_mask.data_ptr<int32_t>(), \
            B, H, N, num_kv_blocks, scale \
        )

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

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
