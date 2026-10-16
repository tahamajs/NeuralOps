#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void w4a16_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& W_q,
    const torch::Tensor& scales,
    const torch::Tensor& zeros,
    int group_size
);

void fp8_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
);
