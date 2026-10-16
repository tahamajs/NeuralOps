#include <torch/extension.h>
torch::Tensor selective_ssm_fwd(const torch::Tensor& x, const torch::Tensor& dt, const torch::Tensor& A, const torch::Tensor& B_vec, const torch::Tensor& C_vec);
