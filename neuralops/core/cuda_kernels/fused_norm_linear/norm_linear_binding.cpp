#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_norm_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const float eps
);

torch::Tensor fused_norm_linear_bwd_dx(
    const torch::Tensor& dout,
    const torch::Tensor& weight,
    const torch::Tensor& x,
    const float eps
);
