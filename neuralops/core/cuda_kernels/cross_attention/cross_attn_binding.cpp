#include <torch/extension.h>

torch::Tensor cross_attention_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v
);
