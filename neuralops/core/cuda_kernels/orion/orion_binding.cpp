#include <torch/extension.h>
torch::Tensor orion_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double temperature);
