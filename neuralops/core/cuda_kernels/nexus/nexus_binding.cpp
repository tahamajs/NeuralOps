#include <torch/extension.h>
torch::Tensor nexus_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double tau);
