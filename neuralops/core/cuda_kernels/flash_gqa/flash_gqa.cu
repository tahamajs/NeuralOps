#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

template<int HEAD_DIM, int BLOCK_M, int BLOCK_N>
__global__ void flash_gqa_warp_kernel(
    float* __restrict__ out,      // [B, H, N, HEAD_DIM]
    const float* __restrict__ q,  // [B, H, N, HEAD_DIM]
    const float* __restrict__ k,  // [B, H_KV, N, HEAD_DIM]
    const float* __restrict__ v,  // [B, H_KV, N, HEAD_DIM]
    const int B,
    const int H,
    const int H_KV,
    const int N,
    const bool is_causal,
    const float scale
) {
    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;
    const int q_idx = q_block_idx * BLOCK_M + warp_id;

    const int kv_h = h / (H / H_KV);

    // Shared memory for staging K and V tiles
    __shared__ float K_smem[BLOCK_N][HEAD_DIM];
    __shared__ float V_smem[BLOCK_N][HEAD_DIM];

    constexpr int ELEMENTS_PER_THREAD = HEAD_DIM / 32;
    float q_local[ELEMENTS_PER_THREAD];

    const int q_offset_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_idx * HEAD_DIM;
    if (q_idx < N) {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            q_local[e] = q[q_offset_base + lane + e * 32];
        }
    } else {
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            q_local[e] = 0.0f;
        }
    }

    // Online softmax state
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[ELEMENTS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
        acc[e] = 0.0f;
    }

    const int num_kv_tiles = (N + BLOCK_N - 1) / BLOCK_N;
    const int tid_in_block = warp_id * 32 + lane;
    const int total_threads_in_block = BLOCK_M * 32;
    const int kv_elements_per_tile = BLOCK_N * HEAD_DIM;

    const int k_base_offset = b * (H_KV * N * HEAD_DIM) + kv_h * (N * HEAD_DIM);

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const int k_start = tile_idx * BLOCK_N;

        if (is_causal && k_start > (q_block_idx * BLOCK_M + (BLOCK_M - 1))) {
            break;
        }

        // Cooperatively load K and V tiles into shared memory
        __syncthreads();
        for (int idx = tid_in_block; idx < kv_elements_per_tile; idx += total_threads_in_block) {
            const int token_in_tile = idx / HEAD_DIM;
            const int dim_in_token = idx % HEAD_DIM;
            const int global_token = k_start + token_in_tile;

            if (global_token < N) {
                const int g_off = k_base_offset + global_token * HEAD_DIM + dim_in_token;
                K_smem[token_in_tile][dim_in_token] = k[g_off];
                V_smem[token_in_tile][dim_in_token] = v[g_off];
            } else {
                K_smem[token_in_tile][dim_in_token] = 0.0f;
                V_smem[token_in_tile][dim_in_token] = 0.0f;
            }
        }
        __syncthreads();

        // Process this KV tile for the warp's query
        if (q_idx < N) {
            const int tokens_in_this_tile = min(BLOCK_N, N - k_start);
            for (int k_t = 0; k_t < tokens_in_this_tile; ++k_t) {
                const int global_k = k_start + k_t;
                if (is_causal && global_k > q_idx) {
                    break;
                }

                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    dot += q_local[e] * K_smem[k_t][lane + e * 32];
                }

                // Butterfly warp reduction
                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                float qk = dot * scale;
                float m_curr = fmaxf(m_prev, qk);
                float alpha = expf(m_prev - m_curr);
                float p = expf(qk - m_curr);

                l_prev = l_prev * alpha + p;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    acc[e] = acc[e] * alpha + p * V_smem[k_t][lane + e * 32];
                }
                m_prev = m_curr;
            }
        }
    }

    // Write normalized output to global memory
    if (q_idx < N && l_prev > 0.0f) {
        const float inv_l = 1.0f / l_prev;
        #pragma unroll
        for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
            out[q_offset_base + lane + e * 32] = acc[e] * inv_l;
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
    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int H_KV = k.size(1);

    constexpr int BLOCK_M = 4;
    constexpr int BLOCK_N = 32;

    dim3 grid((N + BLOCK_M - 1) / BLOCK_M, H, B);
    dim3 block(32, BLOCK_M);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 64) {
        flash_gqa_warp_kernel<64, BLOCK_M, BLOCK_N><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N, is_causal, scale
        );
    } else if (D == 128) {
        flash_gqa_warp_kernel<128, BLOCK_M, BLOCK_N><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N, is_causal, scale
        );
    }
}
