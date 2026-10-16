#include <torch/extension.h>
torch::Tensor decay_linear_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& gamma);
