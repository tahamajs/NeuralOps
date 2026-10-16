#include <torch/extension.h>

torch::Tensor softplus_fwd(
    const torch::Tensor& x,
    const float beta,
    const float threshold
);
