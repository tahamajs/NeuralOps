#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

template<int HEAD_DIM>
__global__ void sliding_window_attn_kernel(
    float* __restrict__ out,      // [B, H, N, D]
    const float* __restrict__ q,  // [B, H, N, D]
    const float* __restrict__ k,  // [B, H, N, D]
    const float* __restrict__ v,  // [B, H, N, D]
    const int B,
    const int H,
    const int N,
    const int window_size,
    const int num_sinks,
    const float scale
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int q_pos = blockIdx.z;

    if (b >= B || h >= H || q_pos >= N) return;

    float q_val[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_val[d] = q[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_pos * HEAD_DIM + d];
    }

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    for (int k_pos = 0; k_pos <= q_pos; ++k_pos) {
        // Condition: either in initial attention sinks or inside sliding window
        const bool is_sink = (k_pos < num_sinks);
        const bool in_window = (k_pos >= (q_pos - window_size + 1));

        if (!is_sink && !in_window) {
            continue;
        }

        float qk = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            qk += q_val[d] * k[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + k_pos * HEAD_DIM + d];
        }
        qk *= scale;

        float m_curr = fmaxf(m_prev, qk);
        float p = expf(qk - m_curr);
        float alpha = expf(m_prev - m_curr);

        l_prev = l_prev * alpha + p;

        for (int d = 0; d < HEAD_DIM; ++d) {
            float v_val = v[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + k_pos * HEAD_DIM + d];
            acc[d] = acc[d] * alpha + p * v_val;
        }
        m_prev = m_curr;
    }

    if (l_prev > 0.0f) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            out[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_pos * HEAD_DIM + d] = acc[d] / l_prev;
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
    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);

    dim3 grid(B, H, N);
    dim3 block(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 64) {
        sliding_window_attn_kernel<64><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, N, window_size, num_sinks, scale
        );
    }
}
