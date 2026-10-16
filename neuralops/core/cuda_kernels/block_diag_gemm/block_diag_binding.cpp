#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor block_diag_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const c10::optional<torch::Tensor>& bias
);
