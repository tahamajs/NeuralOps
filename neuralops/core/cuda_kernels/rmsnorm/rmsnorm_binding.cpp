#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor rmsnorm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& residual,
    const float eps
);
