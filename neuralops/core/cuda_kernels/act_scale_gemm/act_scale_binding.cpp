#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor act_scale_quant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& s_inv,
    const torch::Tensor& w,
    const c10::optional<torch::Tensor>& bias
);
