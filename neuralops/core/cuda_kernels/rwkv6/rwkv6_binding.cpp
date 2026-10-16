#include <torch/extension.h>
torch::Tensor rwkv6_time_mix_fwd(const torch::Tensor& r, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& w, const torch::Tensor& u);
