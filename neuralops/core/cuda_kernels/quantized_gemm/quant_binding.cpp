// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>

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

torch::Tensor w4a16_gemm(
    torch::Tensor A,
    torch::Tensor W_q,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t group_size
) {
    TORCH_CHECK(A.is_cuda() && W_q.is_cuda() && scales.is_cuda() && zeros.is_cuda(),
                "w4a16_gemm: all tensors must be on CUDA");
    TORCH_CHECK(A.dim() == 2 && W_q.dim() == 2 && scales.dim() == 2 && zeros.dim() == 2,
                "w4a16_gemm: all tensors must be 2D");

    const int M = static_cast<int>(A.size(0));
    const int N = static_cast<int>(scales.size(1));

    auto orig_dtype = A.scalar_type();
    auto A_c = (orig_dtype == torch::kFloat32 ? A : A.to(torch::kFloat32)).contiguous();
    auto W_q_c = W_q.contiguous();
    auto scales_c = (scales.scalar_type() == torch::kFloat32 ? scales : scales.to(torch::kFloat32)).contiguous();
    auto zeros_c = (zeros.scalar_type() == torch::kFloat32 ? zeros : zeros.to(torch::kFloat32)).contiguous();

    auto out = torch::zeros({M, N}, A_c.options());
    w4a16_gemm_cuda(out, A_c, W_q_c, scales_c, zeros_c, static_cast<int>(group_size));

    if (orig_dtype != torch::kFloat32) {
        return out.to(orig_dtype);
    }
    return out;
}

torch::Tensor w4a16_gemm_bwd(
    torch::Tensor dout,
    torch::Tensor W_q,
    torch::Tensor scales,
    torch::Tensor zeros,
    int64_t K,
    int64_t group_size
) {
    const int M = static_cast<int>(dout.size(0));
    auto orig_dtype = dout.scalar_type();
    auto dout_c = (orig_dtype == torch::kFloat32 ? dout : dout.to(torch::kFloat32)).contiguous();
    auto W_q_c = W_q.contiguous();
    auto scales_c = (scales.scalar_type() == torch::kFloat32 ? scales : scales.to(torch::kFloat32)).contiguous();
    auto zeros_c = (zeros.scalar_type() == torch::kFloat32 ? zeros : zeros.to(torch::kFloat32)).contiguous();

    auto dA = torch::zeros({M, static_cast<int>(K)}, dout_c.options());
    w4a16_gemm_bwd_cuda(dA, dout_c, W_q_c, scales_c, zeros_c, static_cast<int>(group_size));

    if (orig_dtype != torch::kFloat32) {
        return dA.to(orig_dtype);
    }
    return dA;
}

torch::Tensor fp8_gemm(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor scale_a,
    torch::Tensor scale_b
) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "fp8_gemm: A and B must be CUDA tensors");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "fp8_gemm: A and B must be 2D matrices");

    const int M = static_cast<int>(A.size(0));
    const int N = static_cast<int>(B.size(1));

    auto orig_dtype = A.scalar_type();
    auto A_c = (orig_dtype == torch::kFloat32 ? A : A.to(torch::kFloat32)).contiguous();
    auto B_c = (B.scalar_type() == torch::kFloat32 ? B : B.to(torch::kFloat32)).contiguous();
    auto sa_c = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.to(torch::kFloat32).contiguous() : torch::Tensor();
    auto sb_c = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.to(torch::kFloat32).contiguous() : torch::Tensor();

    auto out = torch::zeros({M, N}, A_c.options());
    fp8_gemm_cuda(out, A_c, B_c, sa_c, sb_c);

    if (orig_dtype != torch::kFloat32) {
        return out.to(orig_dtype);
    }
    return out;
}

std::vector<torch::Tensor> fp8_gemm_bwd(
    torch::Tensor dout,
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor scale_a,
    torch::Tensor scale_b
) {
    const int M = static_cast<int>(A.size(0));
    const int K = static_cast<int>(A.size(1));
    const int N = static_cast<int>(B.size(1));

    auto orig_dtype = A.scalar_type();
    auto dout_c = (dout.scalar_type() == torch::kFloat32 ? dout : dout.to(torch::kFloat32)).contiguous();
    auto A_c = (orig_dtype == torch::kFloat32 ? A : A.to(torch::kFloat32)).contiguous();
    auto B_c = (B.scalar_type() == torch::kFloat32 ? B : B.to(torch::kFloat32)).contiguous();
    auto sa_c = (scale_a.defined() && scale_a.numel() > 0) ? scale_a.to(torch::kFloat32).contiguous() : torch::Tensor();
    auto sb_c = (scale_b.defined() && scale_b.numel() > 0) ? scale_b.to(torch::kFloat32).contiguous() : torch::Tensor();

    auto dA = torch::zeros({M, K}, A_c.options());
    auto dB = torch::zeros({K, N}, B_c.options());

    fp8_gemm_bwd_cuda(dA, dB, dout_c, A_c, B_c, sa_c, sb_c);

    if (orig_dtype != torch::kFloat32) {
        return {dA.to(orig_dtype), dB.to(B.scalar_type())};
    }
    return {dA, dB};
}
