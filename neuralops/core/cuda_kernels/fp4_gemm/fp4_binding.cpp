#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor fp4_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
);

std::vector<torch::Tensor> fp4_gemm_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& x,
    const torch::Tensor& w_packed,
    const torch::Tensor& scales,
    const c10::optional<torch::Tensor>& bias
);
