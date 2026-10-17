#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "_common.cuh"


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
    const int tok_idx = blockIdx.x;
    const int k_idx = blockIdx.y;
    const int d = threadIdx.x;

    if (tok_idx >= num_tokens || k_idx >= top_k) return;

    const int target_slot = (expert_offsets != nullptr) ? expert_offsets[tok_idx * top_k + k_idx] : (tok_idx * top_k + k_idx);
    if (target_slot < 0 || target_slot >= total_assignments) return;

    if ((hidden_dim * sizeof(scalar_t)) % 16 == 0) {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        const int num_vecs = hidden_dim / VEC_SIZE;
        using vec_t = int4;
        const vec_t* __restrict__ src_vec = reinterpret_cast<const vec_t*>(tokens + tok_idx * hidden_dim);
        vec_t* __restrict__ dst_vec = reinterpret_cast<vec_t*>(dispatched_tokens + target_slot * hidden_dim);
        for (int i = d; i < num_vecs; i += blockDim.x) {
            dst_vec[i] = src_vec[i];
        }
    } else {
        for (int i = d; i < hidden_dim; i += blockDim.x) {
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
    if (num_tokens <= 0 || hidden_dim <= 0 || top_k <= 0 || total_assignments <= 0) {
        return;
    }

    dim3 grid(num_tokens, top_k);
    dim3 block(std::min(256, ((hidden_dim * static_cast<int>(tokens.element_size())) % 16 == 0) ? (hidden_dim / (16 / static_cast<int>(tokens.element_size()))) : hidden_dim));
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

    if (lane == 0) {
        float best_val[8];
        int best_indices[8];
        for (int j = 0; j < top_k && j < 8; ++j) {
            best_val[j] = -1e30f;
            best_indices[j] = 0;
        }

        const scalar_t* g_row = gate_logits + tok_idx * num_experts;
        for (int e = 0; e < num_experts; ++e) {
            float val = static_cast<float>(g_row[e]);
            for (int j = 0; j < top_k && j < 8; ++j) {
                if (val > best_val[j]) {
                    for (int m = std::min(top_k - 1, 7); m > j; --m) {
                        best_val[m] = best_val[m - 1];
                        best_indices[m] = best_indices[m - 1];
                    }
                    best_val[j] = val;
                    best_indices[j] = e;
                    break;
                }
            }
        }

        float max_val = best_val[0];
        float sum_exp = 0.0f;
        float weights[8];
        for (int j = 0; j < top_k && j < 8; ++j) {
            weights[j] = expf(best_val[j] - max_val);
            sum_exp += weights[j];
        }
        float inv_sum = 1.0f / (sum_exp > 0.0f ? sum_exp : 1.0f);
        for (int j = 0; j < top_k && j < 8; ++j) {
            routing_weights[tok_idx * top_k + j] = static_cast<scalar_t>(weights[j] * inv_sum);
            topk_indices[tok_idx * top_k + j] = best_indices[j];
        }
    }

    // Parallel vectorized copy of tokens[tok_idx] to dispatched[tok_idx * top_k + j]
    const scalar_t* __restrict__ src_tok = tokens + tok_idx * hidden_dim;
    if ((hidden_dim * sizeof(scalar_t)) % 16 == 0) {
        constexpr int VEC_SIZE = 16 / sizeof(scalar_t);
        const int num_vecs = hidden_dim / VEC_SIZE;
        using vec_t = int4;
        const vec_t* __restrict__ src_vec = reinterpret_cast<const vec_t*>(src_tok);

        for (int i = lane; i < num_vecs; i += 32) {
            const vec_t v = src_vec[i];
            for (int j = 0; j < top_k; ++j) {
                vec_t* __restrict__ dst_vec = reinterpret_cast<vec_t*>(dispatched_tokens + (tok_idx * top_k + j) * hidden_dim);
                dst_vec[i] = v;
            }
        }
    } else {
        for (int i = lane; i < hidden_dim; i += 32) {
            const scalar_t v = src_tok[i];
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

