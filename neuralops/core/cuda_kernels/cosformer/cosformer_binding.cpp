#include <torch/extension.h>

torch::Tensor cosformer_linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const bool is_causal
);
