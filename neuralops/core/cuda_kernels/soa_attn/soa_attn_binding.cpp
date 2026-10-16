#include <torch/extension.h>
torch::Tensor soa_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double alpha);
