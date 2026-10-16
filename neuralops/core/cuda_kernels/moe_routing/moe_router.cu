#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void moe_dispatch_kernel(
    float* __restrict__ dispatched_tokens,     // [total_assignments, hidden_dim]
    const float* __restrict__ tokens,          // [num_tokens, hidden_dim]
    const int64_t* __restrict__ topk_indices,  // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,// [num_tokens, top_k] slot assignments
    const int num_tokens,
    const int hidden_dim,
    const int top_k
) {
    const int tok_idx = blockIdx.x;
    const int k_idx = blockIdx.y;
    const int d = threadIdx.x;

    if (tok_idx >= num_tokens || k_idx >= top_k) return;

    const int target_slot = expert_offsets[tok_idx * top_k + k_idx];
    if (target_slot < 0) return;

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
    int top_k
) {
    dim3 grid(num_tokens, top_k);
    dim3 block(min(1024, hidden_dim));

    moe_dispatch_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        dispatched_tokens.data_ptr<float>(),
        tokens.data_ptr<float>(),
        topk_indices.data_ptr<int64_t>(),
        expert_offsets.data_ptr<int32_t>(),
        num_tokens, hidden_dim, top_k
    );
}
