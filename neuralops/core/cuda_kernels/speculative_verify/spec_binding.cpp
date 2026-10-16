#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> speculative_verify_fwd(
    const torch::Tensor& draft_tokens,
    const torch::Tensor& target_tokens
);
