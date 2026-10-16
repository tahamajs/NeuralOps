#include <torch/extension.h>
torch::Tensor fast_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double eps);
