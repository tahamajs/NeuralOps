// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> mamba2_block_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_mat,
    const torch::Tensor& C_mat,
    const c10::optional<torch::Tensor>& D_val,
    const c10::optional<torch::Tensor>& prev_state
);
