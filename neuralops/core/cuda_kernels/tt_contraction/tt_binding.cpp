#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor tt_contraction_fwd(
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1
);

torch::Tensor tt_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1,
    const c10::optional<torch::Tensor>& bias
);
