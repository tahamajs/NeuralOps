#include <torch/extension.h>
torch::Tensor local_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int window_size);
