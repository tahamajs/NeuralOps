#include <torch/extension.h>
#include <vector>

torch::Tensor diff_attn_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
);

std::vector<torch::Tensor> diff_attn_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
);
