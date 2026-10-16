#include <torch/extension.h>
torch::Tensor galerkin_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double h_step);
