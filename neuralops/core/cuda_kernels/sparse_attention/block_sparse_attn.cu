#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <float.h>

template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void block_sparse_attn_kernel(
    float* __restrict__ out,               // [B, H, N, D]
    const float* __restrict__ q,           // [B, H, N, D]
    const float* __restrict__ k,           // [B, H, N, D]
    const float* __restrict__ v,           // [B, H, N, D]
    const int32_t* __restrict__ block_mask,// [num_q_blocks, num_kv_blocks]
    const int B,
    const int H,
    const int N,
    const int num_kv_blocks,
    const float scale
) {
    const int b = blockIdx.x;
    const int h = blockIdx.y;
    const int q_block_idx = blockIdx.z;
    const int tid = threadIdx.x;

    if (b >= B || h >= H) return;

    const int q_start = q_block_idx * BLOCK_SIZE;
    const int q_end = min(q_start + BLOCK_SIZE, N);

    for (int q_pos = q_start; q_pos < q_end; ++q_pos) {
        float q_val[HEAD_DIM];
        for (int d = 0; d < HEAD_DIM; ++d) {
            q_val[d] = q[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_pos * HEAD_DIM + d];
        }

        float m_prev = -FLT_MAX;
        float l_prev = 0.0f;
        float acc[HEAD_DIM];
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;

        for (int kv_b = 0; kv_b < num_kv_blocks; ++kv_b) {
            // Dynamic skip control flow (not expressible in Triton without full compute)
            if (block_mask[q_block_idx * num_kv_blocks + kv_b] == 0) {
                continue;
            }

            const int k_start = kv_b * BLOCK_SIZE;
            const int k_end = min(k_start + BLOCK_SIZE, N);

            for (int k_pos = k_start; k_pos < k_end; ++k_pos) {
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
        }

        if (l_prev > 0.0f) {
            for (int d = 0; d < HEAD_DIM; ++d) {
                out[b * (H * N * HEAD_DIM) + h * (N * HEAD_DIM) + q_pos * HEAD_DIM + d] = acc[d] / l_prev;
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
    const int B = q.size(0);
    const int H = q.size(1);
    const int N = q.size(2);
    const int D = q.size(3);
    const int num_q_blocks = (N + block_size - 1) / block_size;
    const int num_kv_blocks = block_mask.size(1);

    dim3 grid(B, H, num_q_blocks);
    dim3 block(1);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (D == 64 && block_size == 16) {
        block_sparse_attn_kernel<64, 16><<<grid, block, 0, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k.data_ptr<float>(),
            v.data_ptr<float>(),
            block_mask.data_ptr<int32_t>(),
            B, H, N, num_kv_blocks, scale
        );
    }
}
