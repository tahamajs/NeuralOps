#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

template<typename scalar_t, int HEAD_DIM, int TILE_Q, int BLOCK_N>
__global__ void flash_gqa_kernel(
    scalar_t* __restrict__ out,         // [B, H, N_Q, HEAD_DIM]
    const scalar_t* __restrict__ q,     // [B, H, N_Q, HEAD_DIM]
    const scalar_t* __restrict__ k,     // [B, H_KV, N_KV, HEAD_DIM]
    const scalar_t* __restrict__ v,     // [B, H_KV, N_KV, HEAD_DIM]
    const int B,
    const int H,
    const int H_KV,
    const int N_Q,
    const int N_KV,
    const bool is_causal,
    const float scale
) {
    const int q_tile_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int kv_h = h / (H / H_KV);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;

    const int cur_q_idx = q_tile_idx * TILE_Q + tid;

    // Shared memory for K and V tiles (+1 padding to eliminate bank conflicts)
    __shared__ float K_smem[BLOCK_N][HEAD_DIM + 1];
    __shared__ float V_smem[BLOCK_N][HEAD_DIM + 1];

    // Local registers for this thread's single query vector
    float q_local[HEAD_DIM];
    float acc[HEAD_DIM];
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    const int q_base = b * (H * N_Q * HEAD_DIM) + h * (N_Q * HEAD_DIM) + cur_q_idx * HEAD_DIM;
    if (cur_q_idx < N_Q) {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_local[d] = static_cast<float>(q[q_base + d]);
            acc[d] = 0.0f;
        }
    } else {
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_local[d] = 0.0f;
            acc[d] = 0.0f;
        }
    }

    const int k_base = b * (H_KV * N_KV * HEAD_DIM) + kv_h * (N_KV * HEAD_DIM);
    const int num_kv_tiles = (N_KV + BLOCK_N - 1) / BLOCK_N;
    const int kv_tile_elements = BLOCK_N * HEAD_DIM;

    // Causal horizon check for the tile
    const int max_q_in_tile = (q_tile_idx + 1) * TILE_Q - 1;
    const int max_k_in_tile = max_q_in_tile + (N_KV - N_Q);

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const int k_start = tile_idx * BLOCK_N;
        if (is_causal && k_start > max_k_in_tile) {
            break;
        }

        // Cooperative load of K and V into shared memory
        __syncthreads();
        for (int idx = tid; idx < kv_tile_elements; idx += total_threads) {
            const int kt = idx / HEAD_DIM;
            const int kd = idx % HEAD_DIM;
            const int global_tok = k_start + kt;

            if (global_tok < N_KV) {
                const int g_off = k_base + global_tok * HEAD_DIM + kd;
                K_smem[kt][kd] = static_cast<float>(k[g_off]);
                V_smem[kt][kd] = static_cast<float>(v[g_off]);
            } else {
                K_smem[kt][kd] = 0.0f;
                V_smem[kt][kd] = 0.0f;
            }
        }
        __syncthreads();

        if (cur_q_idx >= N_Q) {
            continue;
        }

        const int valid_keys = min(BLOCK_N, N_KV - k_start);
        const int max_k_for_q = cur_q_idx + (N_KV - N_Q);

        float scores[BLOCK_N];
        float m_tile = -1e30f;

        #pragma unroll
        for (int kt = 0; kt < BLOCK_N; ++kt) {
            const int global_k = k_start + kt;
            if (kt < valid_keys && (!is_causal || global_k <= max_k_for_q)) {
                float dot = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d) {
                    dot += q_local[d] * K_smem[kt][d];
                }
                float s = dot * scale;
                scores[kt] = s;
                m_tile = fmaxf(m_tile, s);
            } else {
                scores[kt] = -1e30f;
            }
        }

        // Tile-level online softmax update (rescale accumulator ONCE per tile)
        if (m_tile > -1e29f) {
            float m_new = fmaxf(m_prev, m_tile);
            float alpha = (m_prev == -1e30f) ? 0.0f : expf(m_prev - m_new);
            m_prev = m_new;
            l_prev *= alpha;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[d] *= alpha;
            }

            #pragma unroll
            for (int kt = 0; kt < BLOCK_N; ++kt) {
                if (scores[kt] > -1e29f) {
                    float p = expf(scores[kt] - m_new);
                    l_prev += p;
                    #pragma unroll
                    for (int d = 0; d < HEAD_DIM; ++d) {
                        acc[d] += p * V_smem[kt][d];
                    }
                }
            }
        }
    }

    // Write final normalized result
    if (cur_q_idx < N_Q) {
        const int out_base = b * (H * N_Q * HEAD_DIM) + h * (N_Q * HEAD_DIM) + cur_q_idx * HEAD_DIM;
        const float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) {
            out[out_base + d] = static_cast<scalar_t>(acc[d] * inv_l);
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

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        q.scalar_type(),
        "flash_gqa_fwd_cuda",
        ([&] {
            if (D == 32) {
                constexpr int TILE_Q = 128;
                constexpr int BLOCK_N = 32;
                dim3 grid((N_Q + TILE_Q - 1) / TILE_Q, H, B);
                dim3 block(32, 4);
                flash_gqa_kernel<scalar_t, 32, TILE_Q, BLOCK_N><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    B, H, H_KV, N_Q, N_KV, is_causal, scale
                );
            } else if (D == 64) {
                constexpr int TILE_Q = 128;
                constexpr int BLOCK_N = 32;
                dim3 grid((N_Q + TILE_Q - 1) / TILE_Q, H, B);
                dim3 block(32, 4);
                flash_gqa_kernel<scalar_t, 64, TILE_Q, BLOCK_N><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    B, H, H_KV, N_Q, N_KV, is_causal, scale
                );
            } else if (D == 128) {
                constexpr int TILE_Q = 64;
                constexpr int BLOCK_N = 16;
                dim3 grid((N_Q + TILE_Q - 1) / TILE_Q, H, B);
                dim3 block(32, 2);
                flash_gqa_kernel<scalar_t, 128, TILE_Q, BLOCK_N><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    B, H, H_KV, N_Q, N_KV, is_causal, scale
                );
            } else if (D == 256) {
                constexpr int TILE_Q = 32;
                constexpr int BLOCK_N = 16;
                dim3 grid((N_Q + TILE_Q - 1) / TILE_Q, H, B);
                dim3 block(32, 1);
                flash_gqa_kernel<scalar_t, 256, TILE_Q, BLOCK_N><<<grid, block, 0, stream>>>(
                    out.data_ptr<scalar_t>(),
                    q.data_ptr<scalar_t>(),
                    k.data_ptr<scalar_t>(),
                    v.data_ptr<scalar_t>(),
                    B, H, H_KV, N_Q, N_KV, is_causal, scale
                );
            }
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
