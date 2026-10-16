#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void sliding_window_attn_fwd_cuda(
    torch::Tensor& out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int window_size,
    int num_sinks,
    float scale
);
