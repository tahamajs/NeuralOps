#include <torch/extension.h>
torch::Tensor hydra_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v);
