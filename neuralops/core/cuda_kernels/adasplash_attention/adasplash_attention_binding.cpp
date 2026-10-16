#include <torch/extension.h>
torch::Tensor adasplash_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, double threshold, int window_size);
