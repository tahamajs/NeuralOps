#include <torch/extension.h>
torch::Tensor routing_attn_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& route_weights, int top_k);
