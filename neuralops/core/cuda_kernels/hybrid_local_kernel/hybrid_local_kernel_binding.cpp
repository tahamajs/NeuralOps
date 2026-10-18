/*
Hybrid Local Kernel Attention C++ Bindings

Author: Taha Majlesi
*/

#include <torch/extension.h>

torch::Tensor hybrid_local_attn_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int radius,
    bool is_causal
);

torch::Tensor hybrid_two_branch_blend_fwd(
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    const torch::Tensor& alpha
);
