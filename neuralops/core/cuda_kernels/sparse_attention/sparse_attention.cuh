#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void block_sparse_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
);

void block_sparse_attn_bwd_cuda(
    torch::Tensor& grad_q,
    torch::Tensor& grad_k,
    torch::Tensor& grad_v,
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& block_mask,
    int block_size,
    float scale
);
