#include <torch/extension.h>
torch::Tensor spectral_mamba_fwd(const torch::Tensor& x, const torch::Tensor& freq_coeffs, double threshold);
