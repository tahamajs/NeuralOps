#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> liquid_attention_fwd(
    const torch::Tensor& x,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const c10::optional<torch::Tensor>& prev_state
);
