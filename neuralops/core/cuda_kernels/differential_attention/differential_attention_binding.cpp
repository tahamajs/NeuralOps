// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>

torch::Tensor diff_attn_fwd(
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
);

std::vector<torch::Tensor> diff_attn_bwd(
    const torch::Tensor& grad_out,
    const torch::Tensor& q1,
    const torch::Tensor& k1,
    const torch::Tensor& q2,
    const torch::Tensor& k2,
    const torch::Tensor& v,
    double lambda_val
);
