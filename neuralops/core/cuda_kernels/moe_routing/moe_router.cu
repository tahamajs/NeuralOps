#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "_common.cuh"

// ═══════════════════════════════════════════════════════════════════════
// MoE Dispatch Kernel: Vectorized 128-bit Token Scattering
// ═══════════════════════════════════════════════════════════════════════
template<typename scalar_t>
__launch_bounds__(256)
__global__ void moe_dispatch_kernel(
    scalar_t* __restrict__ dispatched_tokens,     // [total_assignments, hidden_dim]
    const scalar_t* __restrict__ tokens,          // [num_tokens, hidden_dim]
    const int32_t* __restrict__ topk_indices,     // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,   // [num_tokens, top_k] slot assignments (optional)
    const int num_tokens,
    const int hidden_dim,
    const int top_k,
    const int total_assignments
) {
    const int tok_idx = blockIdx.y;
    const int k_idx = blockIdx.z;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;

    if (tok_idx >= num_tokens || k_idx >= top_k) return;

    const int target_slot = (expert_offsets != nullptr) ? expert_offsets[tok_idx * top_k + k_idx] : (tok_idx * top_k + k_idx);
    if (target_slot < 0 || target_slot >= total_assignments) return;

    if ((hidden_dim * sizeof(scalar_t)) % 16 == 0) {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        const int num_vecs = hidden_dim / VEC_SIZE;
        using vec_t = int4;
        const vec_t* __restrict__ src_vec = reinterpret_cast<const vec_t*>(tokens + tok_idx * hidden_dim);
        vec_t* __restrict__ dst_vec = reinterpret_cast<vec_t*>(dispatched_tokens + target_slot * hidden_dim);
        for (int i = d; i < num_vecs; i += blockDim.x * gridDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (int i = d; i < hidden_dim; i += blockDim.x * gridDim.x) {
            dispatched_tokens[target_slot * hidden_dim + i] = tokens[tok_idx * hidden_dim + i];
        }
    }
}

void moe_dispatch_cuda(
    torch::Tensor& dispatched_tokens,
    const torch::Tensor& tokens,
    const torch::Tensor& topk_indices,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int total_assignments
) {
    CHECK_INPUT(dispatched_tokens);
    CHECK_INPUT(tokens);
    if (topk_indices.defined() && topk_indices.numel() > 0) {
        CHECK_INPUT(topk_indices);
    }
    if (expert_offsets.defined() && expert_offsets.numel() > 0) {
        CHECK_INPUT(expert_offsets);
    }

    if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0 || total_assignments <= 0) {
        return;
    }

    const int block_size = 256;
    const int effective_d = ((hidden_dim * static_cast<int>(tokens.element_size())) % 16 == 0) ? (hidden_dim / (16 / static_cast<int>(tokens.element_size()))) : hidden_dim;
    const int grid_x = (effective_d + block_size - 1) / block_size;
    dim3 grid(grid_x, num_tokens, top_k);
    dim3 block(block_size);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    const int32_t* expert_offsets_ptr = (expert_offsets.defined() && expert_offsets.numel() > 0) ? expert_offsets.data_ptr<int32_t>() : nullptr;
    const int32_t* topk_indices_ptr = (topk_indices.defined() && topk_indices.numel() > 0) ? topk_indices.data_ptr<int32_t>() : nullptr;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        tokens.scalar_type(),
        "moe_dispatch_cuda",
        ([&] {
            moe_dispatch_kernel<scalar_t><<<grid, block, 0, stream>>>(
                dispatched_tokens.data_ptr<scalar_t>(),
                tokens.data_ptr<scalar_t>(),
                topk_indices_ptr,
                expert_offsets_ptr,
                num_tokens,
                hidden_dim,
                top_k,
                total_assignments
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// ═══════════════════════════════════════════════════════════════════════
// Bitonic MoE Router: Parallel Warp-Shuffle Top-K Selection & Softmax
// ═══════════════════════════════════════════════════════════════════════
template<typename scalar_t>
__launch_bounds__(128)
__global__ void moe_fused_dispatch_kernel(
    scalar_t* __restrict__ dispatched_tokens, // [num_tokens * top_k, hidden_dim]
    scalar_t* __restrict__ routing_weights,   // [num_tokens, top_k]
    int32_t* __restrict__ topk_indices,       // [num_tokens, top_k]
    const scalar_t* __restrict__ tokens,      // [num_tokens, hidden_dim]
    const scalar_t* __restrict__ gate_logits, // [num_tokens, num_experts]
    const int num_tokens,
    const int hidden_dim,
    const int num_experts,
    const int top_k
) {
    const int tok_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    if (tok_idx >= num_tokens) return;

    // Bitonic Top-K selection across 32 lanes
    float val = -1e30f;
    int idx = 0;

    const scalar_t* g_row = gate_logits + tok_idx * num_experts;
    if (lane < num_experts) {
        val = to_float(g_row[lane]);
        idx = lane;
    }

    // If num_experts > 32, fold subsequent experts into the 32 lanes
    for (int e = lane + 32; e < num_experts; e += 32) {
        float cand_val = to_float(g_row[e]);
        if (cand_val > val) {
            val = cand_val;
            idx = e;
        }
    }

    // 32-element Bitonic Sorting Network in pure register shuffle space
    #pragma unroll
    for (int size = 2; size <= 32; size *= 2) {
        #pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2) {
            float other_val = __shfl_xor_sync(0xffffffff, val, stride);
            int other_idx = __shfl_xor_sync(0xffffffff, idx, stride);
            bool dir = ((lane & size) == 0); // descending for first half
            bool swap = dir ? (val < other_val) : (val > other_val);
            if (swap) {
                val = other_val;
                idx = other_idx;
            }
        }
    }

    // Lanes 0 to (top_k - 1) now hold the sorted top-k values and expert indices!
    float max_val = __shfl_sync(0xffffffff, val, 0);
    float exp_val = (lane < top_k) ? expf(val - max_val) : 0.0f;

    // Warp-level sum reduction for softmax normalizer
    float sum_exp = exp_val;
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        sum_exp += __shfl_down_sync(0xffffffff, sum_exp, offset);
    }
    sum_exp = __shfl_sync(0xffffffff, sum_exp, 0);
    float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);

    if (lane < top_k) {
        routing_weights[tok_idx * top_k + lane] = from_float<scalar_t>(exp_val * inv_sum);
        topk_indices[tok_idx * top_k + lane] = idx;
    }

    // Parallel vectorized 128-bit streaming copy of tokens[tok_idx] to dispatched[tok_idx * top_k + j]
    const scalar_t* __restrict__ src_tok = tokens + tok_idx * hidden_dim;
    if ((hidden_dim * sizeof(scalar_t)) % 16 == 0) {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        const int num_vecs = hidden_dim / VEC_SIZE;
        using vec_t = int4;
        const vec_t* __restrict__ src_vec = reinterpret_cast<const vec_t*>(src_tok);

        for (int i = lane; i < num_vecs; i += 32) {
            const vec_t v = src_vec[i];
            #pragma unroll
            for (int j = 0; j < top_k; ++j) {
                vec_t* __restrict__ dst_vec = reinterpret_cast<vec_t*>(dispatched_tokens + (tok_idx * top_k + j) * hidden_dim);
                dst_vec[i] = v;
            }
        }
    } else {
        for (int i = lane; i < hidden_dim; i += 32) {
            const scalar_t v = src_tok[i];
            #pragma unroll
            for (int j = 0; j < top_k; ++j) {
                dispatched_tokens[(tok_idx * top_k + j) * hidden_dim + i] = v;
            }
        }
    }
}

void moe_fused_dispatch_cuda(
    torch::Tensor& dispatched_tokens,
    torch::Tensor& routing_weights,
    torch::Tensor& topk_indices,
    const torch::Tensor& tokens,
    const torch::Tensor& gate_logits,
    int num_tokens,
    int hidden_dim,
    int num_experts,
    int top_k
) {
    CHECK_INPUT(dispatched_tokens);
    CHECK_INPUT(routing_weights);
    CHECK_INPUT(topk_indices);
    CHECK_INPUT(tokens);
    CHECK_INPUT(gate_logits);

    TORCH_CHECK(top_k <= 32, "moe_fused_dispatch_cuda: top_k=", top_k, " exceeds warp width");

    if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0 || num_experts <= 0) {
        return;
    }

    dim3 block(32, 4); // 4 warps per block = 4 tokens per block
    dim3 grid((num_tokens + 3) / 4);
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        tokens.scalar_type(),
        "moe_fused_dispatch_cuda",
        ([&] {
            moe_fused_dispatch_kernel<scalar_t><<<grid, block, 0, stream>>>(
                dispatched_tokens.data_ptr<scalar_t>(),
                routing_weights.data_ptr<scalar_t>(),
                topk_indices.data_ptr<int32_t>(),
                tokens.data_ptr<scalar_t>(),
                gate_logits.data_ptr<scalar_t>(),
                num_tokens,
                hidden_dim,
                num_experts,
                top_k
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
