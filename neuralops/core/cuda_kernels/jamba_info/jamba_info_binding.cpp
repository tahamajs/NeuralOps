#include <torch/extension.h>
torch::Tensor jamba_info_fwd(const torch::Tensor& x, const torch::Tensor& bottleneck_weights, double eps);
