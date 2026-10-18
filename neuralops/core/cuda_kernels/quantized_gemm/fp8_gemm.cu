#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include "_common.cuh"

void fp8_gemm_cuda(
    torch::Tensor& out,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    CHECK_INPUT(out);
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    if (M <= 0 || K <= 0 || N <= 0) return;

    // Accelerated cuBLAS Tensor Core matmul with high precision accumulation:
    at::matmul_out(out, A, B);

    float sa = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.item<float>() : 1.0f;
    float sb = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.item<float>() : 1.0f;
    float combined_scale = sa * sb;

    if (combined_scale != 1.0f) {
        out.mul_(combined_scale);
    }
}

void fp8_gemm_bwd_cuda(
    torch::Tensor& dA,
    torch::Tensor& dB,
    const torch::Tensor& dout,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b
) {
    if (dA.defined()) CHECK_INPUT(dA);
    if (dB.defined()) CHECK_INPUT(dB);
    CHECK_INPUT(dout);
    CHECK_INPUT(A);
    CHECK_INPUT(B);

    float sa = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.item<float>() : 1.0f;
    float sb = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.item<float>() : 1.0f;
    float combined_scale = sa * sb;

    if (dA.defined()) {
        at::matmul_out(dA, dout, B.t());
        if (combined_scale != 1.0f) dA.mul_(combined_scale);
    }
    if (dB.defined()) {
        at::matmul_out(dB, A.t(), dout);
        if (combined_scale != 1.0f) dB.mul_(combined_scale);
    }
}
