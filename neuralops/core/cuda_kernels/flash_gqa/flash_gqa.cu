#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_gqa_warp_kernel(
    float* __restrict__ out,      // [B, H, N_Q, HEAD_DIM]
    const float* __restrict__ q,  // [B, H, N_Q, HEAD_DIM]
    const float* __restrict__ k,  // [B, H_KV, N_KV, HEAD_DIM]
    const float* __restrict__ v,  // [B, H_KV, N_KV, HEAD_DIM]
    const int B,
    const int H,
    const int H_KV,
    const int N_Q,
    const int N_KV,
    const bool is_causal,
    const float scale
) {
    constexpr int NUM_WARPS = 4;
    constexpr int QUERIES_PER_WARP = BLOCK_M / NUM_WARPS;
    constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / 32;

    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    const int kv_h = h / (H / H_KV);

    // Shared memory for staging K and V tiles with +1 padding to eliminate bank conflicts
    __shared__ float K_smem[BLOCK_N][HEAD_DIM + 1];
    __shared__ float V_smem[BLOCK_N][HEAD_DIM + 1];

    // Local registers for queries in this warp
    float q_local[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float acc[QUERIES_PER_WARP][ELEMENTS_PER_THREAD];
    float m_prev[QUERIES_PER_WARP];
    float l_prev[QUERIES_PER_WARP];
    int q_indices[QUERIES_PER_WARP];

    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        q_indices[qi] = q_block_idx * BLOCK_M + warp_id * QUERIES_PER_WARP + qi;
        m_prev[qi] = -FLT_MAX;
        l_prev[qi] = 0.0f;
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            acc[qi][e] = 0.0f;
            if (q_indices[qi] < N_Q) {
                const int q_off = b * (H * N_Q * HEAD_DIM) + h * (N_Q * HEAD_DIM) + q_indices[qi] * HEAD_DIM + lane + e * 32;
                q_local[qi][e] = q[q_off];
            } else {
                q_local[qi][e] = 0.0f;
            }
        }
    }

    const int num_kv_tiles = (N_KV + BLOCK_N - 1) / BLOCK_N;
    const int tid_in_block = warp_id * 32 + lane;
    const int total_threads = NUM_WARPS * 32;
    const int kv_elements_per_tile = BLOCK_N * HEAD_DIM;
    const int k_base_offset = b * (H_KV * N_KV * HEAD_DIM) + kv_h * (N_KV * HEAD_DIM);

    // Causal horizon check for the block
    const int max_block_q = q_block_idx * BLOCK_M + BLOCK_M - 1;
    const int max_block_k = max_block_q + (N_KV - N_Q);

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const int k_start = tile_idx * BLOCK_N;

        if (is_causal && k_start > max_block_k) {
            break;
        }

        // Cooperatively load K and V tile into SMEM
        __syncthreads();
        for (int idx = tid_in_block; idx < kv_elements_per_tile; idx += total_threads) {
            const int tok_in_tile = idx / HEAD_DIM;
            const int dim_in_tok = idx % HEAD_DIM;
            const int global_tok = k_start + tok_in_tile;

            if (global_tok < N_KV) {
                const int g_off = k_base_offset + global_tok * HEAD_DIM + dim_in_tok;
                K_smem[tok_in_tile][dim_in_tok] = k[g_off];
                V_smem[tok_in_tile][dim_in_tok] = v[g_off];
            } else {
                K_smem[tok_in_tile][dim_in_tok] = 0.0f;
                V_smem[tok_in_tile][dim_in_tok] = 0.0f;
            }
        }
        __syncthreads();

        const int tokens_in_tile = min(BLOCK_N, N_KV - k_start);

        // Process all queries assigned to this warp using tile-level online softmax
        #pragma unroll
        for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
            const int cur_q_idx = q_indices[qi];
            if (cur_q_idx >= N_Q) continue;

            const int max_k_for_query = cur_q_idx + (N_KV - N_Q);
            if (is_causal && max_k_for_query < 0) continue;

            float qk_tile[BLOCK_N];
            float m_tile = -FLT_MAX;

            for (int kt = 0; kt < tokens_in_tile; ++kt) {
                const int global_k = k_start + kt;
                if (is_causal && global_k > max_k_for_query) {
                    qk_tile[kt] = -FLT_MAX;
                    continue;
                }

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
                qk_tile[kt] = qk;
                m_tile = fmaxf(m_tile, qk);
            }

            // Tile-level online softmax update (rescale accumulator ONCE per tile)
            float m_new = fmaxf(m_prev[qi], m_tile);
            float alpha = (m_prev[qi] == -FLT_MAX) ? 0.0f : expf(m_prev[qi] - m_new);
            m_prev[qi] = m_new;
            l_prev[qi] *= alpha;

            #pragma unroll
            for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                acc[qi][e] *= alpha;
            }

            // Accumulate V values for the active tile
            for (int kt = 0; kt < tokens_in_tile; ++kt) {
                if (qk_tile[kt] == -FLT_MAX) continue;
                float p = expf(qk_tile[kt] - m_new);
                l_prev[qi] += p;

                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    acc[qi][e] += p * V_smem[kt][lane + e * 32];
                }
            }
        }
    }

    // Write results to global memory
    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_indices[qi];
        if (cur_q_idx < N_Q) {
            const int out_base = b * (H * N_Q * HEAD_DIM) + h * (N_Q * HEAD_DIM) + cur_q_idx * HEAD_DIM;
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

void flash_gqa_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    bool is_causal,
    float scale
) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int N_Q = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));
    const int H_KV = static_cast<int>(k.size(1));
    const int N_KV = static_cast<int>(k.size(2));

    if (B <= 0 || H <= 0 || N_Q <= 0 || D <= 0 || N_KV <= 0) {
        return;
    }

    constexpr int BLOCK_M = 16;
    constexpr int BLOCK_N = 16;

    dim3 grid((N_Q + BLOCK_M - 1) / BLOCK_M, H, B);
    dim3 block(32, 4); // 4 warps = 128 threads
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 32) {
        flash_gqa_warp_kernel<32, BLOCK_M, BLOCK_N><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N_Q, N_KV, is_causal, scale
        );
    } else if (D == 64) {
        flash_gqa_warp_kernel<64, BLOCK_M, BLOCK_N><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N_Q, N_KV, is_causal, scale
        );
    } else if (D == 128) {
        flash_gqa_warp_kernel<128, BLOCK_M, BLOCK_N><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N_Q, N_KV, is_causal, scale
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
