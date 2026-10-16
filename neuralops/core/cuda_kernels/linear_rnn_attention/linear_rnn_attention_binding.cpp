#include <torch/extension.h>
torch::Tensor linear_rnn_fwd(const torch::Tensor& x, const torch::Tensor& a, const torch::Tensor& b);
