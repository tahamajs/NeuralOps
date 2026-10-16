#include <torch/extension.h>
torch::Tensor spectraformer_fwd(const torch::Tensor& x, const torch::Tensor& cos_basis, const torch::Tensor& sin_basis);
