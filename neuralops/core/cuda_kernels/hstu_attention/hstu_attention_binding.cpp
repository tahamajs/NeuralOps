#include <torch/extension.h>
torch::Tensor hstu_fwd(const torch::Tensor& u, const torch::Tensor& v, const torch::Tensor& decay, double eps);
