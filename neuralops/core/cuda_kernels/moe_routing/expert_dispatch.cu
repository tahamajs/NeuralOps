#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template<typename scalar_t>
__global__ void moe_gather_fast_kernel(
    scalar_t* __restrict__ out_tokens,               // [num_tokens, hidden_dim]
    const scalar_t* __restrict__ expert_outputs,     // [total_assignments, hidden_dim]
    const scalar_t* __restrict__ routing_weights,    // [num_tokens, top_k]
    const int num_tokens,
    const int hidden_dim,
    const int top_k
) {
    const int tok_idx = blockIdx.x * blockDim.y + threadIdx.y;
    const int lane = threadIdx.x;

    if (tok_idx >= num_tokens) return;

    float w[8];
    for (int j = 0; j < top_k && j < 8; ++j) {
        w[j] = static_cast<float>(routing_weights[tok_idx * top_k + j]);
    }

    scalar_t* __restrict__ dst = out_tokens + tok_idx * hidden_dim;

    if ((hidden_dim * sizeof(scalar_t)) % 16 == 0 && std::is_same<scalar_t, float>::value) {
        const int num_vecs = hidden_dim / 4;
        for (int i = lane; i < num_vecs; i += 32) {
            float4 acc = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int j = 0; j < top_k && j < 8; ++j) {
                const float4 in_vec = reinterpret_cast<const float4*>(expert_outputs + (tok_idx * top_k + j) * hidden_dim)[i];
                acc.x += w[j] * in_vec.x;
                acc.y += w[j] * in_vec.y;
                acc.z += w[j] * in_vec.z;
                acc.w += w[j] * in_vec.w;
            }
            reinterpret_cast<float4*>(dst)[i] = acc;
        }
    } else {
        for (int i = lane; i < hidden_dim; i += 32) {
            float acc = 0.0f;
            for (int j = 0; j < top_k && j < 8; ++j) {
                acc += w[j] * static_cast<float>(expert_outputs[(tok_idx * top_k + j) * hidden_dim + i]);
            }
            dst[i] = static_cast<scalar_t>(acc);
        }
    }
}

template<typename scalar_t>
__global__ void moe_gather_kernel(
    scalar_t* __restrict__ out_tokens,               // [num_tokens, hidden_dim]
    const scalar_t* __restrict__ expert_outputs,     // [total_assignments, hidden_dim]
    const scalar_t* __restrict__ routing_weights,    // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,      // [num_tokens, top_k] slot assignments (optional)
    const int num_tokens,
    const int hidden_dim,
    const int top_k,
    const int total_assignments
) {
    const int tok_idx = blockIdx.x;
    const int d = threadIdx.x;

    if (tok_idx >= num_tokens) return;

    for (int i = d; i < hidden_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < top_k; ++k_idx) {
            const int slot = (expert_offsets != nullptr) ? expert_offsets[tok_idx * top_k + k_idx] : (tok_idx * top_k + k_idx);
            if (slot >= 0 && slot < total_assignments) {
                const float weight = static_cast<float>(routing_weights[tok_idx * top_k + k_idx]);
                acc += weight * static_cast<float>(expert_outputs[slot * hidden_dim + i]);
            }
        }
        out_tokens[tok_idx * hidden_dim + i] = static_cast<scalar_t>(acc);
    }
}

void moe_gather_cuda(
    torch::Tensor& out_tokens,
    const torch::Tensor& expert_outputs,
    const torch::Tensor& routing_weights,
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

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    const int32_t* expert_offsets_ptr = (expert_offsets.defined() && expert_offsets.numel() > 0) ? expert_offsets.data_ptr<int32_t>() : nullptr;

    if (expert_offsets_ptr == nullptr && top_k <= 8) {
        dim3 block(32, 4);
        dim3 grid((num_tokens + 3) / 4);
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            expert_outputs.scalar_type(),
            "moe_gather_fast_cuda",
            ([&] {
                moe_gather_fast_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    out_tokens.data_ptr<scalar_t>(),
                    expert_outputs.data_ptr<scalar_t>(),
                    routing_weights.data_ptr<scalar_t>(),
                    num_tokens,
                    hidden_dim,
                    top_k
                );
            })
        );
    } else {
        dim3 grid(num_tokens);
        dim3 block(std::min(256, hidden_dim));
        AT_DISPATCH_FLOATING_TYPES_AND2(
            at::ScalarType::Half,
            at::ScalarType::BFloat16,
            expert_outputs.scalar_type(),
            "moe_gather_cuda",
            ([&] {
                moe_gather_kernel<scalar_t><<<grid, block, 0, stream>>>(
                    out_tokens.data_ptr<scalar_t>(),
                    expert_outputs.data_ptr<scalar_t>(),
                    routing_weights.data_ptr<scalar_t>(),
                    expert_offsets_ptr,
                    num_tokens,
                    hidden_dim,
                    top_k,
                    total_assignments
                );
            })
        );
    }

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

