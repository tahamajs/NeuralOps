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

torch::Tensor w4a16_gemm(
    torch::Tensor A,
    torch::Tensor W_q,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size
) {
    const int M = A.size(0);
    const int N = scales.size(1);
    auto out = torch::empty({M, N}, A.options());
    w4a16_gemm_cuda(out, A, W_q, scales, zeros, static_cast<int>(group_size));
    return out;
}

torch::Tensor fp8_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor scale_a,
    torch::Tensor scale_b
) {
    const int M = A.size(0);
    const int N = B.size(1);
    auto out = torch::empty({M, N}, A.options());
    fp8_gemm_cuda(out, A, B, scale_a, scale_b);
    return out;
}
