#include <torch/extension.h>

torch::Tensor kronecker_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& bias
);
