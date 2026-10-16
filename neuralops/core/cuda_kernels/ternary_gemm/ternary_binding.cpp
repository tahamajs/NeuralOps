#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor ternary_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
);
