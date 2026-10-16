#include <torch/extension.h>

void moe_dispatch_cuda(
    torch::Tensor& dispatched_tokens,
    const torch::Tensor& tokens,
    const torch::Tensor& topk_indices,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k
);

void moe_gather_cuda(
    torch::Tensor& out_tokens,
    const torch::Tensor& expert_outputs,
    const torch::Tensor& routing_weights,
    const torch::Tensor& topk_indices,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k
);

torch::Tensor moe_dispatch(
    torch::Tensor tokens,
    torch::Tensor topk_indices,
    torch::Tensor expert_offsets,
    int64_t total_slots
) {
    const int num_tokens = tokens.size(0);
    const int hidden_dim = tokens.size(1);
    const int top_k = topk_indices.size(1);

    auto dispatched = torch::zeros({total_slots, hidden_dim}, tokens.options());
    moe_dispatch_cuda(dispatched, tokens, topk_indices, expert_offsets, num_tokens, hidden_dim, top_k);
    return dispatched;
}

torch::Tensor moe_gather(
    torch::Tensor expert_outputs,
    torch::Tensor routing_weights,
    torch::Tensor topk_indices,
    torch::Tensor expert_offsets
) {
    const int num_tokens = routing_weights.size(0);
    const int top_k = routing_weights.size(1);
    const int hidden_dim = expert_outputs.size(1);

    auto out = torch::zeros({num_tokens, hidden_dim}, expert_outputs.options());
    moe_gather_cuda(out, expert_outputs, routing_weights, topk_indices, expert_offsets, num_tokens, hidden_dim, top_k);
    return out;
}
