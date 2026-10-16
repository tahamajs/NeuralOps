#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor causal_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias,
    const bool silu_activation
);
