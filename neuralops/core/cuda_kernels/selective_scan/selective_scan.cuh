#pragma once

#include <cuda_runtime.h>
#include <torch/extension.h>

void selective_scan_fwd_cuda(
    torch::Tensor& out,
    torch::Tensor& last_state,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D
);

void selective_scan_bwd_cuda(
    torch::Tensor& du,
    torch::Tensor& ddelta,
    torch::Tensor& dA,
    torch::Tensor& dB,
    torch::Tensor& dC,
    torch::Tensor& dD,
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& C,
    const torch::Tensor& D,
    const torch::Tensor& dout
);
