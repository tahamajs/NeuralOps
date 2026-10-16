#include <torch/extension.h>
torch::Tensor hymba_transx_fwd(const torch::Tensor& x, const torch::Tensor& meta_tokens, const torch::Tensor& cross_gate);
