// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <float.h>
#include "_common.cuh"

#define WARP_SIZE 32

template<int HEAD_DIM, int BLOCK_SIZE>
__launch_bounds__(32)
__global__ void paged_attention_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int max_num_blocks_per_seq,
    const int num_heads,
    const int num_kv_heads,
    const int num_total_blocks,
    const float scale
) {
    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    constexpr int ELEMS_PER_THREAD = HEAD_DIM / WARP_SIZE;
    const int out_seq_head_base = seq_idx * num_heads * HEAD_DIM + head_idx * HEAD_DIM;

    const int context_len = context_lens[seq_idx];
    if (context_len <= 0) {
        #pragma unroll
        for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
            out[out_seq_head_base + tid + e * WARP_SIZE] = 0.0f;
        }
        return;
    }

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int num_blocks = min(max_num_blocks_per_seq, (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE);

    float q_reg[ELEMS_PER_THREAD];
    float acc[ELEMS_PER_THREAD];
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        q_reg[e] = q[out_seq_head_base + tid + e * WARP_SIZE];
        acc[e] = 0.0f;
    }

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int b = 0; b < num_blocks; ++b) {
        const int physical_block_idx = seq_block_table[b];
        if (physical_block_idx < 0 || physical_block_idx >= num_total_blocks) {
            continue;
        }

        const int curr_block_tokens = min(BLOCK_SIZE, context_len - b * BLOCK_SIZE);
        const int kv_block_base = physical_block_idx * (num_kv_heads * HEAD_DIM * BLOCK_SIZE)
                                  + kv_head_idx * (HEAD_DIM * BLOCK_SIZE);

        for (int tok = 0; tok < curr_block_tokens; ++tok) {
            float qk_partial = 0.0f;
            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE;
                const int k_offset = kv_block_base + d * BLOCK_SIZE + tok;
                qk_partial += q_reg[e] * k_cache[k_offset];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
            }

            const float qk = qk_partial * scale;

            const float m_curr = fmaxf(m_prev, qk);
            const float p = expf(qk - m_curr);
            const float alpha = expf(m_prev - m_curr);

            l_prev = l_prev * alpha + p;

            #pragma unroll
            for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
                const int d = tid + e * WARP_SIZE;
                const int v_offset = kv_block_base + d * BLOCK_SIZE + tok;
                acc[e] = acc[e] * alpha + p * v_cache[v_offset];
            }
            m_prev = m_curr;
        }
    }

    const float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    #pragma unroll
    for (int e = 0; e < ELEMS_PER_THREAD; ++e) {
        out[out_seq_head_base + tid + e * WARP_SIZE] = acc[e] * inv_l;
    }
}

__launch_bounds__(32)
__global__ void paged_attention_dynamic_kernel(
    float* __restrict__ out,
    const float* __restrict__ q,
    const float* __restrict__ k_cache,
    const float* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const int max_num_blocks_per_seq,
    const int num_heads,
    const int num_kv_heads,
    const int num_total_blocks,
    const int head_dim,
    const int block_size,
    const float scale
) {
    extern __shared__ float smem[];
    float* q_shared = smem;
    float* acc_shared = smem + head_dim;

    const int seq_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int tid = threadIdx.x;

    const int out_seq_head_base = seq_idx * num_heads * head_dim + head_idx * head_dim;
    const int context_len = context_lens[seq_idx];

    if (context_len <= 0) {
        for (int d = tid; d < head_dim; d += blockDim.x) {
            out[out_seq_head_base + d] = 0.0f;
        }
        return;
    }

    for (int d = tid; d < head_dim; d += blockDim.x) {
        q_shared[d] = q[out_seq_head_base + d];
        acc_shared[d] = 0.0f;
    }
    __syncthreads();

    const int kv_head_idx = head_idx / (num_heads / num_kv_heads);
    const int num_blocks = min(max_num_blocks_per_seq, (context_len + block_size - 1) / block_size);

    float m_prev = -FLT_MAX;
    float l_prev = 0.0f;

    const int* seq_block_table = block_tables + seq_idx * max_num_blocks_per_seq;

    for (int b = 0; b < num_blocks; ++b) {
        const int physical_block_idx = seq_block_table[b];
        if (physical_block_idx < 0 || physical_block_idx >= num_total_blocks) {
            continue;
        }

        const int curr_block_tokens = min(block_size, context_len - b * block_size);
        const int kv_block_base = physical_block_idx * (num_kv_heads * head_dim * block_size)
                                  + kv_head_idx * (head_dim * block_size);

        for (int tok = 0; tok < curr_block_tokens; ++tok) {
            float qk_partial = 0.0f;
            for (int d = tid; d < head_dim; d += blockDim.x) {
                qk_partial += q_shared[d] * k_cache[kv_block_base + d * block_size + tok];
            }

            #pragma unroll
            for (int offset = 16; offset > 0; offset /= 2) {
                qk_partial += __shfl_xor_sync(0xffffffff, qk_partial, offset);
            }

            const float qk = qk_partial * scale;
            const float m_curr = fmaxf(m_prev, qk);
            const float p = expf(qk - m_curr);
            const float alpha = expf(m_prev - m_curr);

            l_prev = l_prev * alpha + p;

            for (int d = tid; d < head_dim; d += blockDim.x) {
                acc_shared[d] = acc_shared[d] * alpha + p * v_cache[kv_block_base + d * block_size + tok];
            }
            m_prev = m_curr;
        }
    }
    __syncthreads();

    const float inv_l = (l_prev > 0.0f) ? (1.0f / l_prev) : 0.0f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        out[out_seq_head_base + d] = acc_shared[d] * inv_l;
    }
}

#define LAUNCH_PAGED_ATTN(HD, BS) \
    paged_attention_kernel<HD, BS><<<grid, block, 0, stream>>>( \
        out.data_ptr<float>(), \
        q.data_ptr<float>(), \
        k_cache.data_ptr<float>(), \
        v_cache.data_ptr<float>(), \
        block_tables.data_ptr<int>(), \
        context_lens.data_ptr<int>(), \
        max_num_blocks_per_seq, \
        num_heads, \
        num_kv_heads, \
        num_total_blocks, \
        scale \
    )

void paged_attention_v1_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache,
    const torch::Tensor& block_tables,
    const torch::Tensor& context_lens,
    int block_size,
    float scale
) {
    CHECK_INPUT(out);
    CHECK_INPUT(q);
    CHECK_INPUT(k_cache);
    CHECK_INPUT(v_cache);
    CHECK_INPUT(block_tables);
    CHECK_INPUT(context_lens);

    const int num_seqs = q.size(0);
    if (num_seqs == 0) return;

    const int num_heads = q.size(1);
    const int head_dim = q.size(2);
    const int num_kv_heads = k_cache.size(1);
    const int num_total_blocks = k_cache.size(0);
    const int max_num_blocks_per_seq = block_tables.size(1);

    dim3 grid(num_seqs, num_heads);
    dim3 block(WARP_SIZE);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    if (head_dim == 32 && block_size == 8) {
        LAUNCH_PAGED_ATTN(32, 8);
    } else if (head_dim == 32 && block_size == 16) {
        LAUNCH_PAGED_ATTN(32, 16);
    } else if (head_dim == 32 && block_size == 32) {
        LAUNCH_PAGED_ATTN(32, 32);
    } else if (head_dim == 64 && block_size == 8) {
        LAUNCH_PAGED_ATTN(64, 8);
    } else if (head_dim == 64 && block_size == 16) {
        LAUNCH_PAGED_ATTN(64, 16);
    } else if (head_dim == 64 && block_size == 32) {
        LAUNCH_PAGED_ATTN(64, 32);
    } else if (head_dim == 128 && block_size == 8) {
        LAUNCH_PAGED_ATTN(128, 8);
    } else if (head_dim == 128 && block_size == 16) {
        LAUNCH_PAGED_ATTN(128, 16);
    } else if (head_dim == 128 && block_size == 32) {
        LAUNCH_PAGED_ATTN(128, 32);
    } else {
        const size_t smem_bytes = 2 * head_dim * sizeof(float);
        paged_attention_dynamic_kernel<<<grid, block, smem_bytes, stream>>>(
            out.data_ptr<float>(),
            q.data_ptr<float>(),
            k_cache.data_ptr<float>(),
            v_cache.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            max_num_blocks_per_seq,
            num_heads,
            num_kv_heads,
            num_total_blocks,
            head_dim,
            block_size,
            scale
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
