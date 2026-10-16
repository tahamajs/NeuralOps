#include <torch/extension.h>
torch::Tensor spectra_v2_fwd(const torch::Tensor& x, const torch::Tensor& freq_bands, double threshold);
