#include <torch/extension.h>

torch::Tensor multi_scale_retention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& gammas
);
