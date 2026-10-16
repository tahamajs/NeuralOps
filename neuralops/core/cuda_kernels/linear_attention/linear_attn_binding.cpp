#include <torch/extension.h>

torch::Tensor linear_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
);
