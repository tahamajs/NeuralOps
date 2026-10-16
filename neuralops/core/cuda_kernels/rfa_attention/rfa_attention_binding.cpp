#include <torch/extension.h>
torch::Tensor rfa_fwd(const torch::Tensor& phi_q, const torch::Tensor& phi_k, const torch::Tensor& v);
