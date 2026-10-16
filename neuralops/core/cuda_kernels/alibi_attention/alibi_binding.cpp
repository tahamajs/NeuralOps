#include <torch/extension.h>

torch::Tensor alibi_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& slopes,
    const bool is_causal
);
