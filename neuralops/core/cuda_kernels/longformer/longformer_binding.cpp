#include <torch/extension.h>
torch::Tensor longformer_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int window_size, int num_global_tokens);
