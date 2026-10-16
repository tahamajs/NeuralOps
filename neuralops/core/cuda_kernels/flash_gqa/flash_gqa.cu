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
    constexpr int NUM_VECS = HEAD_DIM / 4;
    const int q_tile_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int kv_h = h / (H / H_KV);
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    const int total_threads = blockDim.x * blockDim.y;
    const int cur_q_idx = q_tile_idx * TILE_Q + tid;

    // 16-byte aligned Shared memory for K and V tiles (+1 float4 padding to eliminate bank conflicts)
    __shared__ float4 K_smem[BLOCK_N][NUM_VECS + 1];
    __shared__ float4 V_smem[BLOCK_N][NUM_VECS + 1];

    float4 q_vec[NUM_VECS];
    float4 acc_vec[NUM_VECS];
    float m_prev = -1e30f;
    float l_prev = 0.0f;

    const int q_base = b * (H * N_Q * HEAD_DIM) + h * (N_Q * HEAD_DIM) + cur_q_idx * HEAD_DIM;
    if (cur_q_idx < N_Q) {
        #pragma unroll
        for (int d = 0; d < NUM_VECS; ++d) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                q_vec[d] = reinterpret_cast<const float4*>(q + q_base)[d];
            } else {
                q_vec[d] = make_float4(
                    static_cast<float>(q[q_base + d * 4 + 0]),
                    static_cast<float>(q[q_base + d * 4 + 1]),
                    static_cast<float>(q[q_base + d * 4 + 2]),
                    static_cast<float>(q[q_base + d * 4 + 3])
                );
            }
            acc_vec[d] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    } else {
        #pragma unroll
        for (int d = 0; d < NUM_VECS; ++d) {
            q_vec[d] = make_float4(0.f, 0.f, 0.f, 0.f);
            acc_vec[d] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }

    const int k_base = b * (H_KV * N_KV * HEAD_DIM) + kv_h * (N_KV * HEAD_DIM);
    const int num_kv_tiles = (N_KV + BLOCK_N - 1) / BLOCK_N;
    constexpr int TOTAL_VECS_PER_TILE = BLOCK_N * NUM_VECS;

    for (int tile_idx = 0; tile_idx < num_kv_tiles; ++tile_idx) {
        const int k_start = tile_idx * BLOCK_N;
        if (is_causal && k_start > cur_q_idx + (N_KV - N_Q)) {
            break;
        }

        // Cooperative load of K and V into 128-bit shared memory
        __syncthreads();
        #pragma unroll
        for (int idx = tid; idx < TOTAL_VECS_PER_TILE; idx += total_threads) {
            const int kt = idx / NUM_VECS;
            const int kd = idx % NUM_VECS;
            const int global_tok = k_start + kt;

            if (global_tok < N_KV) {
                const int g_off = k_base + global_tok * HEAD_DIM + kd * 4;
                if constexpr (std::is_same<scalar_t, float>::value) {
                    K_smem[kt][kd] = reinterpret_cast<const float4*>(k + g_off)[0];
                    V_smem[kt][kd] = reinterpret_cast<const float4*>(v + g_off)[0];
                } else {
                    K_smem[kt][kd] = make_float4(
                        static_cast<float>(k[g_off + 0]),
                        static_cast<float>(k[g_off + 1]),
                        static_cast<float>(k[g_off + 2]),
                        static_cast<float>(k[g_off + 3])
                    );
                    V_smem[kt][kd] = make_float4(
                        static_cast<float>(v[g_off + 0]),
                        static_cast<float>(v[g_off + 1]),
                        static_cast<float>(v[g_off + 2]),
                        static_cast<float>(v[g_off + 3])
                    );
                }
            } else {
                K_smem[kt][kd] = make_float4(0.f, 0.f, 0.f, 0.f);
                V_smem[kt][kd] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        if (cur_q_idx >= N_Q) continue;

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
                for (int d = 0; d < NUM_VECS; ++d) {
                    const float4 k_v = K_smem[kt][d];
                    dot += q_vec[d].x * k_v.x + q_vec[d].y * k_v.y + q_vec[d].z * k_v.z + q_vec[d].w * k_v.w;
                }
                const float s = dot * scale;
                scores[kt] = s;
                m_tile = fmaxf(m_tile, s);
            } else {
                scores[kt] = -1e30f;
            }
        }

        // Tile-level online softmax
        if (m_tile > -1e29f) {
            const float m_new = fmaxf(m_prev, m_tile);
            const float alpha = (m_prev == -1e30f) ? 0.0f : expf(m_prev - m_new);
            m_prev = m_new;
            l_prev *= alpha;

            #pragma unroll
            for (int d = 0; d < NUM_VECS; ++d) {
                acc_vec[d].x *= alpha;
                acc_vec[d].y *= alpha;
                acc_vec[d].z *= alpha;
                acc_vec[d].w *= alpha;
            }

            #pragma unroll
            for (int kt = 0; kt < BLOCK_N; ++kt) {
                if (scores[kt] > -1e29f) {
                    const float p = expf(scores[kt] - m_new);
                    l_prev += p;
                    #pragma unroll
                    for (int d = 0; d < NUM_VECS; ++d) {
                        const float4 v_v = V_smem[kt][d];
                        acc_vec[d].x += p * v_v.x;
                        acc_vec[d].y += p * v_v.y;
                        acc_vec[d].z += p * v_v.z;
                        acc_vec[d].w += p * v_v.w;
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
        for (int d = 0; d < NUM_VECS; ++d) {
            if constexpr (std::is_same<scalar_t, float>::value) {
                reinterpret_cast<float4*>(out + out_base)[d] = make_float4(
                    acc_vec[d].x * inv_l,
                    acc_vec[d].y * inv_l,
                    acc_vec[d].z * inv_l,
                    acc_vec[d].w * inv_l
                );
            } else {
                out[out_base + d * 4 + 0] = static_cast<scalar_t>(acc_vec[d].x * inv_l);
                out[out_base + d * 4 + 1] = static_cast<scalar_t>(acc_vec[d].y * inv_l);
                out[out_base + d * 4 + 2] = static_cast<scalar_t>(acc_vec[d].z * inv_l);
                out[out_base + d * 4 + 3] = static_cast<scalar_t>(acc_vec[d].w * inv_l);
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
