#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void moe_gather_kernel(
    float* __restrict__ out_tokens,               // [num_tokens, hidden_dim]
    const float* __restrict__ expert_outputs,     // [total_assignments, hidden_dim]
    const float* __restrict__ routing_weights,    // [num_tokens, top_k]
    const int32_t* __restrict__ expert_offsets,   // [num_tokens, top_k] slot assignments
    const int num_tokens,
    const int hidden_dim,
    const int top_k
) {
    const int tok_idx = blockIdx.x;
    const int d = threadIdx.x;

    if (tok_idx >= num_tokens) return;

    for (int i = d; i < hidden_dim; i += blockDim.x) {
        float acc = 0.0f;
        for (int k_idx = 0; k_idx < top_k; ++k_idx) {
            const int slot = expert_offsets[tok_idx * top_k + k_idx];
            if (slot >= 0) {
                const float weight = routing_weights[tok_idx * top_k + k_idx];
                acc += weight * expert_outputs[slot * hidden_dim + i];
            }
        }
        out_tokens[tok_idx * hidden_dim + i] = acc;
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
    int top_k
) {
    dim3 grid(num_tokens);
    dim3 block(min(1024, hidden_dim));

    moe_gather_kernel<<<grid, block, 0, at::cuda::getCurrentCUDAStream()>>>(
        out_tokens.data_ptr<float>(),
        expert_outputs.data_ptr<float>(),
        routing_weights.data_ptr<float>(),
        expert_offsets.data_ptr<int32_t>(),
        num_tokens, hidden_dim, top_k
    );
}
