#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>

template<int HEAD_DIM, int BLOCK_M>
__launch_bounds__(128)
__global__ void sliding_window_attn_warp_kernel(
    float* __restrict__ out,      // [B, H, N, HEAD_DIM]
    const float* __restrict__ q,  // [B, H, N, HEAD_DIM]
    const float* __restrict__ k,  // [B, H, N, HEAD_DIM]
    const float* __restrict__ v,  // [B, H, N, HEAD_DIM]
    const int B,
    const int H,
    const int N,
    const int window_size,
    const int num_sinks,
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

    if (b >= B || h >= H) return;

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
            if (q_indices[qi] < N) {
                const int q_off = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_indices[qi] * HEAD_DIM + lane + e * 32;
                q_local[qi][e] = q[q_off];
            } else {
                q_local[qi][e] = 0.0f;
            }
        }
    }

    const int kv_head_base = b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM);

    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_indices[qi];
        if (cur_q_idx >= N) continue;

        const int sink_end = min(cur_q_idx + 1, num_sinks);
        const int win_start = max(sink_end, cur_q_idx - window_size + 1);
        const int win_end = cur_q_idx + 1;

        // Helper lambda or macro to process range [start, end)
        auto process_range = [&](int start_pos, int end_pos) {
            for (int k_pos = start_pos; k_pos < end_pos; ++k_pos) {
                const int k_off = kv_head_base + k_pos * HEAD_DIM;
                float dot = 0.0f;
                #pragma unroll
                for (int e = 0; e < ELEMENTS_PER_THREAD; ++e) {
                    dot += q_local[qi][e] * k[k_off + lane + e * 32];
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
                    acc[qi][e] = acc[qi][e] * alpha + p * v[k_off + lane + e * 32];
                }
                m_prev[qi] = m_curr;
            }
        };

        // 1. Initial Attention Sinks
        if (sink_end > 0) {
            process_range(0, sink_end);
        }

        // 2. Sliding Window (disjoint from sinks)
        if (win_start < win_end) {
            process_range(win_start, win_end);
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

template<int BLOCK_M>
__launch_bounds__(128)
__global__ void sliding_window_attn_generic_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const int B,
    const int H,
    const int N,
    const int D,
    const int window_size,
    const int num_sinks,
    const float scale
) {
    constexpr int NUM_WARPS = 4;
    constexpr int QUERIES_PER_WARP = BLOCK_M / NUM_WARPS;

    const int q_block_idx = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    const int lane = threadIdx.x;
    const int warp_id = threadIdx.y;

    if (b >= B || h >= H) return;

    const int kv_head_base = b * (H * N * D) + h * (N * D);

    #pragma unroll
    for (int qi = 0; qi < QUERIES_PER_WARP; ++qi) {
        const int cur_q_idx = q_block_idx * BLOCK_M + warp_id * QUERIES_PER_WARP + qi;
        if (cur_q_idx >= N) continue;

        const int q_off = kv_head_base + cur_q_idx * D;
        const int sink_end = min(cur_q_idx + 1, num_sinks);
        const int win_start = max(sink_end, cur_q_idx - window_size + 1);
        const int win_end = cur_q_idx + 1;

        float m_prev = -FLT_MAX;
        float l_prev = 0.0f;

        // In generic kernel, compute reduction per active token
        auto process_range = [&](int start_pos, int end_pos) {
            for (int k_pos = start_pos; k_pos < end_pos; ++k_pos) {
                const int k_off = kv_head_base + k_pos * D;
                float dot = 0.0f;
                for (int d = lane; d < D; d += 32) {
                    dot += q[q_off + d] * k[k_off + d];
                }

                #pragma unroll
                for (int mask = 16; mask > 0; mask /= 2) {
                    dot += __shfl_xor_sync(0xffffffff, dot, mask);
                }

                float qk = dot * scale;
                float m_curr = fmaxf(m_prev, qk);
                float alpha = expf(m_prev - m_curr);
                float p = expf(qk - m_curr);

                l_prev = l_prev * alpha + p;
                for (int d = lane; d < D; d += 32) {
                    out[q_off + d] = out[q_off + d] * alpha + p * v[k_off + d];
                }
                m_prev = m_curr;
            }
        };

        if (sink_end > 0) process_range(0, sink_end);
        if (win_start < win_end) process_range(win_start, win_end);

        if (l_prev > 0.0f) {
            for (int d = lane; d < D; d += 32) {
                out[q_off + d] /= l_prev;
            }
        }
    }
}

void sliding_window_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int num_sinks,
    float scale
) {
    const int B = static_cast<int>(q.size(0));
    const int H = static_cast<int>(q.size(1));
    const int N = static_cast<int>(q.size(2));
    const int D = static_cast<int>(q.size(3));

    if (B <= 0 || H <= 0 || N <= 0 || D <= 0) return;

    constexpr int BLOCK_M = 16;
    dim3 grid((N + BLOCK_M - 1) / BLOCK_M, H, B);
    dim3 block(32, 4); // 4 warps = 128 threads
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 32) {
        sliding_window_attn_warp_kernel<32, BLOCK_M><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            B, H, N, window_size, num_sinks, scale
        );
    } else if (D == 64) {
        sliding_window_attn_warp_kernel<64, BLOCK_M><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            B, H, N, window_size, num_sinks, scale
        );
    } else if (D == 128) {
        sliding_window_attn_warp_kernel<128, BLOCK_M><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            B, H, N, window_size, num_sinks, scale
        );
    } else if (D == 256) {
        sliding_window_attn_warp_kernel<256, BLOCK_M><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            B, H, N, window_size, num_sinks, scale
        );
    } else {
        sliding_window_attn_generic_kernel<BLOCK_M><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(), q.data_ptr<float>(), k.data_ptr<float>(), v.data_ptr<float>(),
            B, H, N, D, window_size, num_sinks, scale
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
