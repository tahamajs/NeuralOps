#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template<typename scalar_t>
__global__ void moe_dispatch_kernel(
    scalar_t* __restrict__ dispatched_tokens,     // [total_assignments, hidden_dim]
    const scalar_t* __restrict__ tokens,          // [num_tokens, hidden_dim]
    const int32_t* __restrict__ topk_indices,     // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,   // [num_tokens, top_k] slot assignments
    const int num_tokens,
    const int hidden_dim,
    const int top_k,
    const int total_assignments
) {
    const int tok_idx = blockIdx.x;
    const int k_idx = blockIdx.y;
    const int d = threadIdx.x;

    if (tok_idx >= num_tokens || k_idx >= top_k) return;

    const int target_slot = expert_offsets[tok_idx * top_k + k_idx];
    if (target_slot < 0 || target_slot >= total_assignments) return;

    for (int i = d; i < hidden_dim; i += blockDim.x) {
        dispatched_tokens[target_slot * hidden_dim + i] = tokens[tok_idx * hidden_dim + i];
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
    dim3 block(min(1024, hidden_dim));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        tokens.scalar_type(),
        "moe_dispatch_cuda",
        ([&] {
            moe_dispatch_kernel<scalar_t><<<grid, block, 0, stream>>>(
                dispatched_tokens.data_ptr<scalar_t>(),
                tokens.data_ptr<scalar_t>(),
                topk_indices.data_ptr<int32_t>(),
                expert_offsets.data_ptr<int32_t>(),
                num_tokens,
                hidden_dim,
                top_k,
                total_assignments
            );
        })
    );

    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
