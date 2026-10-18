#include <torch/extension.h>
#include <vector>

torch::Tensor cross_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
);

std::vector<torch::Tensor> cross_attention_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
);
