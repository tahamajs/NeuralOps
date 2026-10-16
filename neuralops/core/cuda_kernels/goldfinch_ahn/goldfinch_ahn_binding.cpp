#include <torch/extension.h>
torch::Tensor goldfinch_ahn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& beta);
