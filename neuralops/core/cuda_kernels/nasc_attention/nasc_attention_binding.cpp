#include <torch/extension.h>
torch::Tensor nasc_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& decay);
