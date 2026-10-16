#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <float.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

template<int HEAD_DIM, int BLOCK_SIZE>
__global__ void paged_attention_kernel(
    float* __restrict__ out,                // [num_seqs, num_heads, head_dim]
    const float* __restrict__ q,            // [num_seqs, num_heads, head_dim]
    const float* __restrict__ k_cache,      // [num_blocks, num_kv_heads, head_dim, block_size]
    const float* __restrict__ v_cache,      // [num_blocks, num_kv_heads, head_dim, block_size]
    const int* __restrict__ block_tables,   // [num_seqs, max_num_blocks_per_seq]
    const int* __restrict__ context_lens,   // [num_seqs]
    const int max_num_blocks_per_seq,
    const int num_heads,
    const int num_kv_heads,
    const float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) return;

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Load Q into registers / shared memory
    __shared__ float q_shared[HEAD_DIM];
    for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
        q_shared[d] = q[seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d];
    }
    __syncthreads();

    // Online softmax tracking variables
    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;
    float acc[HEAD_DIM];
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        acc[d] = 0.0f;
    }

    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int b = 0; b < num_blocks; ++b) {
        const int physical_block_idx = seq_block_table[b];
        const int curr_block_tokens = min(BLOCK_SIZE, context_len - b * BLOCK_SIZE);

        for (int tok = 0; tok < curr_block_tokens; ++tok) {
            // Compute Q . K
            float qk = 0.0f;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                // k_cache shape: [num_blocks, num_kv_heads, head_dim, block_size]
                const int k_offset = physical_block_idx * (num_kv_heads * HEAD_DIM * BLOCK_SIZE)
                                     + kv_head_idx * (HEAD_DIM * BLOCK_SIZE)
                                     + d * BLOCK_SIZE
                                     + tok;
                qk += q_shared[d] * k_cache[k_offset];
            }
            qk *= scale;

            // Online softmax update
            float m_curr = fmaxf(m_prev, qk);
            float p = expf(qk - m_curr);
            float alpha = expf(m_prev - m_curr);

            l_prev = l_prev * alpha + p;

            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                const int v_offset = physical_block_idx * (num_kv_heads * HEAD_DIM * BLOCK_SIZE)
                                     + kv_head_idx * (HEAD_DIM * BLOCK_SIZE)
                                     + d * BLOCK_SIZE
                                     + tok;
                float v_val = v_cache[v_offset];
                acc[d] = acc[d] * alpha + p * v_val;
            }
            m_prev = m_curr;
        }
    }

    // Write normalized output to HBM
    if (l_prev > 0.0f) {
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            out[seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM + d] = acc[d] / l_prev;
        }
    }
}

void paged_attention_v1_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int block_size,
    int max_context_len,
    float scale
) {
    const int num_seqs = q.size(0);
    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int max_num_blocks_per_seq = block_tables.size(1);

    dim3 grid(num_seqs, num_heads);
    dim3 block(32);

    if (head_dim == 64 && block_size == 16) {
        paged_attention_kernel<64, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            max_num_blocks_per_seq,
            num_heads,
            num_kv_heads,
            scale
        );
    } else if (head_dim == 128 && block_size == 16) {
        paged_attention_kernel<128, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            max_num_blocks_per_seq,
            num_heads,
            num_kv_heads,
            scale
        );
    } else {
        // Dynamic fallback configuration
        paged_attention_kernel<64, 16><<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            max_num_blocks_per_seq,
            num_heads,
            num_kv_heads,
            scale
        );
    }
}
