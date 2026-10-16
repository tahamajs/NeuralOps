#include <torch/extension.h>
torch::Tensor upi_mamba_fwd(const torch::Tensor& x, const torch::Tensor& upi_tokens, double blend_weight);
