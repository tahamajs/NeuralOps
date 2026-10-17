#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor fused_dropout_add_fwd(
    const torch::Tensor& x,
    const torch::Tensor& residual,
    const float p,
    const bool training,
    const uint64_t seed
);
