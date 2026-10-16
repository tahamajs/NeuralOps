#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void kv_cache_copy_cow_cuda(
    torch::Tensor& dst_k,
    torch::Tensor& dst_v,
    const torch::Tensor& src_k,
    const torch::Tensor& src_v,
    const torch::Tensor& block_mapping,
    int num_blocks,
    int block_size,
    int num_heads,
    int head_dim
);
