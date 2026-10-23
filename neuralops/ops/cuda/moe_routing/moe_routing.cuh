// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#pragma once

#include <cuda_runtime.h>
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

void moe_dispatch_bwd_cuda(
    torch::Tensor& grad_tokens,
    const torch::Tensor& grad_dispatched,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int total_assignments
);

void moe_gather_bwd_cuda(
    torch::Tensor& grad_expert_outputs,
    torch::Tensor& grad_routing_weights,
    const torch::Tensor& grad_out,
    const torch::Tensor& expert_outputs,
    const torch::Tensor& routing_weights,
    const torch::Tensor& expert_offsets,
    int num_tokens,
    int hidden_dim,
    int top_k,
    int total_assignments
);



