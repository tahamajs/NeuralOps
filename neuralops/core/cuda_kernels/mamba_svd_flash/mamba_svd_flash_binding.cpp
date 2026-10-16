#include <torch/extension.h>
torch::Tensor mamba_svd_flash_fwd(const torch::Tensor& u, const torch::Tensor& svd_u, const torch::Tensor& svd_vt, const torch::Tensor& decay);
