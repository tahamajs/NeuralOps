// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

void moe_dispatch_cuda(
    torch::Tensor& dispatched_tokens,
    const torch::Tensor& tokens,
    const torch::Tensor& topk_indices,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int total_assignments
);

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
);

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
);


torch::Tensor moe_dispatch(
    torch::Tensor tokens,
    torch::Tensor topk_indices,
    torch::Tensor expert_offsets,
    int64_t total_slots
) {
    TORCH_CHECK(tokens.is_cuda() && topk_indices.is_cuda() && expert_offsets.is_cuda(),
                "moe_dispatch: all input tensors must be on CUDA");
    TORCH_CHECK(tokens.dim() == 2, "moe_dispatch: expected 2D tokens tensor [num_tokens, hidden_dim]");
    TORCH_CHECK(topk_indices.dim() == 2 && expert_offsets.dim() == 2,
                "moe_dispatch: topk_indices and expert_offsets must be 2D tensors");
    TORCH_CHECK(tokens.size(0) == topk_indices.size(0) && topk_indices.sizes() == expert_offsets.sizes(),
                "moe_dispatch: tokens, topk_indices, and expert_offsets dimension mismatch");
    TORCH_CHECK(total_slots >= 0, "moe_dispatch: total_slots must be non-negative");

    const int num_tokens = static_cast<int>(tokens.size(0));
    const int hidden_dim = static_cast<int>(tokens.size(1));
    const int top_k = static_cast<int>(topk_indices.size(1));
    const int total_assignments = static_cast<int>(total_slots);

    auto dispatched = torch::zeros({total_slots, hidden_dim}, tokens.options());
    if (num_tokens == 0 || hidden_dim == 0 || total_assignments == 0) {
        return dispatched;
    }

    auto tokens_c = tokens.contiguous();
    auto topk_indices_c = (topk_indices.scalar_type() == torch::kInt32) ? topk_indices.contiguous() : topk_indices.to(torch::kInt32).contiguous();
    auto expert_offsets_c = (expert_offsets.defined() && expert_offsets.numel() > 0) ? ((expert_offsets.scalar_type() == torch::kInt32) ? expert_offsets.contiguous() : expert_offsets.to(torch::kInt32).contiguous()) : expert_offsets;

    moe_dispatch_cuda(
        dispatched,
        tokens_c,
        topk_indices_c,
        expert_offsets_c,
        num_tokens,
        hidden_dim,
        top_k,
        total_assignments
    );

    return dispatched;
}

torch::Tensor moe_gather(
    torch::Tensor expert_outputs,
    torch::Tensor routing_weights,
    torch::Tensor topk_indices,
    torch::Tensor expert_offsets
) {
    TORCH_CHECK(expert_outputs.is_cuda() && routing_weights.is_cuda() &&
                topk_indices.is_cuda(),
                "moe_gather: expert_outputs, routing_weights, and topk_indices must be on CUDA");
    TORCH_CHECK(expert_outputs.dim() == 2 && routing_weights.dim() == 2,
                "moe_gather: expert_outputs and routing_weights must be 2D");
    TORCH_CHECK(routing_weights.sizes() == topk_indices.sizes(),
                "moe_gather: routing_weights and topk_indices shape mismatch");

    const bool has_offsets = expert_offsets.defined() && expert_offsets.numel() > 0;
    if (has_offsets) {
        TORCH_CHECK(expert_offsets.is_cuda() && expert_offsets.sizes() == routing_weights.sizes(),
                    "moe_gather: expert_offsets shape mismatch");
    }

    const int num_tokens = static_cast<int>(routing_weights.size(0));
    const int top_k = static_cast<int>(routing_weights.size(1));
    const int hidden_dim = static_cast<int>(expert_outputs.size(1));
    const int total_assignments = static_cast<int>(expert_outputs.size(0));

    auto out = torch::empty({num_tokens, hidden_dim}, expert_outputs.options());
    if (num_tokens == 0 || hidden_dim == 0) {
        return out;
    }

    auto expert_outputs_c = expert_outputs.contiguous();
    auto routing_weights_c = routing_weights.to(expert_outputs.scalar_type()).contiguous();
    auto topk_indices_c = (topk_indices.scalar_type() == torch::kInt32) ? topk_indices.contiguous() : topk_indices.to(torch::kInt32).contiguous();
    auto expert_offsets_c = has_offsets ? ((expert_offsets.scalar_type() == torch::kInt32) ? expert_offsets.contiguous() : expert_offsets.to(torch::kInt32).contiguous()) : torch::Tensor();

    moe_gather_cuda(
        out,
        expert_outputs_c,
        routing_weights_c,
        topk_indices_c,
        expert_offsets_c,
        num_tokens,
        hidden_dim,
        top_k,
        total_assignments
    );

    return out;
}


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> moe_fused_dispatch(
    torch::Tensor tokens,
    torch::Tensor gate_logits,
    int64_t top_k
) {
    TORCH_CHECK(tokens.is_cuda() && gate_logits.is_cuda(),
                "moe_fused_dispatch: tokens and gate_logits must be CUDA tensors");
    TORCH_CHECK(tokens.dim() == 2 && gate_logits.dim() == 2,
                "moe_fused_dispatch: expected 2D tensors");
    TORCH_CHECK(tokens.size(0) == gate_logits.size(0),
                "moe_fused_dispatch: batch size mismatch between tokens and gate_logits");
    TORCH_CHECK(top_k > 0 && top_k <= gate_logits.size(1),
                "moe_fused_dispatch: invalid top_k");

    const int num_tokens = static_cast<int>(tokens.size(0));
    const int hidden_dim = static_cast<int>(tokens.size(1));
    const int num_experts = static_cast<int>(gate_logits.size(1));
    const int k = static_cast<int>(top_k);

    auto dispatched = torch::empty({num_tokens * k, hidden_dim}, tokens.options());
    auto routing_weights = torch::empty({num_tokens, k}, tokens.options());
    auto topk_indices = torch::empty({num_tokens, k}, tokens.options().dtype(torch::kInt32));

    if (num_tokens == 0 || hidden_dim == 0) {
        return std::make_tuple(dispatched, routing_weights, topk_indices);
    }

    auto tokens_c = tokens.contiguous();
    auto gate_logits_c = gate_logits.to(tokens.scalar_type()).contiguous();

    moe_fused_dispatch_cuda(
        dispatched,
        routing_weights,
        topk_indices,
        tokens_c,
        gate_logits_c,
        num_tokens,
        hidden_dim,
        num_experts,
        k
    );

    return std::make_tuple(dispatched, routing_weights, topk_indices);
}

