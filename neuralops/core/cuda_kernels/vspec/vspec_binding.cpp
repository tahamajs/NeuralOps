#include <torch/extension.h>
torch::Tensor vspec_fwd(const torch::Tensor& x, const torch::Tensor& spec_gate, const torch::Tensor& decay, double eps);
