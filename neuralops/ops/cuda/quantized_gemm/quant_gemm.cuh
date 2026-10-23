// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

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

void w4a16_gemm_bwd_cuda(
    torch::Tensor& dA,
    const torch::Tensor& dout,
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

void fp8_gemm_bwd_cuda(
    torch::Tensor& dA,
    torch::Tensor& dB,
    const torch::Tensor& dout,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
);
