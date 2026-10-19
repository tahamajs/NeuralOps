// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> recurrent_hgrn2_step_fwd(
    const torch::Tensor& x_t,
    const torch::Tensor& g_t,
    const torch::Tensor& alpha_t,
    torch::Tensor& h_prev
);
