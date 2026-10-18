#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor fp8_gemm_fwd(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const c10::optional<torch::Tensor>& bias,
    const torch::ScalarType& out_dtype
);
