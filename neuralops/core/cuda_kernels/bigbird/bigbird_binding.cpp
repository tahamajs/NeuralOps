#include <torch/extension.h>
torch::Tensor bigbird_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int window_size, int num_global_tokens, int num_random_blocks);
