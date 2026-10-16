#include <torch/extension.h>
torch::Tensor griffin_ssd_fwd(const torch::Tensor& x, const torch::Tensor& r, const torch::Tensor& decay);
