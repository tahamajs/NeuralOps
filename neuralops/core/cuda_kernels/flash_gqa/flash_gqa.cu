#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

template<int HEAD_DIM>
__global__ void flash_gqa_fwd_kernel(
    float* __restrict__ out,      // [B, H, N, D]
    const float* __restrict__ q,  // [B, H, N, D]
    const float* __restrict__ k,  // [B, H_KV, N, D]
    const float* __restrict__ v,  // [B, H_KV, N, D]
    const int B,
    const int H,
    const int H_KV,
    const int N,
    const bool is_causal,
    const float scale
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int q_idx = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= B || h >= H || q_idx >= N) return;

    const int kv_h = h / (H / H_KV);

    // Load Q token into registers
    float q_reg[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) {
        q_reg[d] = q[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_idx * HEAD_DIM + d];
    }

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

    const int max_k = is_causal ? (q_idx + 1) : N;

    for (int k_idx = 0; k_idx < max_k; ++k_idx) {
        float qk = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            qk += q_reg[d] * k[b * (H_KV * N * HEAD_DIM) + kv_h * (N * HEAD_DIM) + k_idx * HEAD_DIM + d];
        }
        qk *= scale;

        float m_curr = fmaxf(m_prev, qk);
        float p = expf(qk - m_curr);
        float alpha = expf(m_prev - m_curr);

        l_prev = l_prev * alpha + p;

        for (int d = 0; d < HEAD_DIM; ++d) {
            float v_val = v[b * (H_KV * N * HEAD_DIM) + kv_h * (N * HEAD_DIM) + k_idx * HEAD_DIM + d];
            acc[d] = acc[d] * alpha + p * v_val;
        }
        m_prev = m_curr;
    }

    if (l_prev > 0.0f) {
        for (int d = 0; d < HEAD_DIM; ++d) {
            out[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_idx * HEAD_DIM + d] = acc[d] / l_prev;
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

    dim3 grid(B, H, N);
    dim3 block(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 64) {
        flash_gqa_fwd_kernel<64><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            B, H, H_KV, N, is_causal, scale
        );
    }
}
