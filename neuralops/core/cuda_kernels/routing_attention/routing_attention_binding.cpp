#include <torch/extension.h>
#include <vector>

torch::Tensor routing_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& route_weights,
    int top_k
);

std::vector<torch::Tensor> routing_attn_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& route_weights,
    int top_k
);
