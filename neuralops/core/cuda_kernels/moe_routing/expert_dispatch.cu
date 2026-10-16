#pragma once
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>

template<typename scalar_t>
__global__ void moe_gather_kernel(
    scalar_t* __restrict__ out_tokens,               // [num_tokens, hidden_dim]
    const scalar_t* __restrict__ expert_outputs,     // [total_assignments, hidden_dim]
    const scalar_t* __restrict__ routing_weights,    // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,      // [num_tokens, top_k] slot assignments
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
            const int slot = expert_offsets[tok_idx * top_k + k_idx];
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

    dim3 grid(num_tokens);
    dim3 block(min(1024, hidden_dim));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

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
