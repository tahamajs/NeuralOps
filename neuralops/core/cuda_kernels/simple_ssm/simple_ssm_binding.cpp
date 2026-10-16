#include <torch/extension.h>
torch::Tensor simple_ssm_fwd(const torch::Tensor& x, const torch::Tensor& decay, const torch::Tensor& scale);
