#include <vector>

std::vector<torch::Tensor> spectral_shrinkage_fwd(const torch::Tensor& x_re, const torch::Tensor& x_im, const torch::Tensor& w_re, const torch::Tensor& w_im, const torch::Tensor& tau);
