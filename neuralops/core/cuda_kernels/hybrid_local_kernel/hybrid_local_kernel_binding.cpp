// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

/*
Hybrid Local Kernel Attention C++ Bindings

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

torch::Tensor hybrid_three_branch_blend_fwd(
    const torch::Tensor& b0,
    const torch::Tensor& b1,
    const torch::Tensor& b2,
    const torch::Tensor& alpha
);
