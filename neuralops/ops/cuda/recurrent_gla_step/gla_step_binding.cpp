// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <vector>
#include <c10/util/Optional.h>

std::vector<torch::Tensor> recurrent_gla_step_fwd(
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    torch::Tensor& s_state,
    const torch::Tensor& alpha_t
);

std::vector<torch::Tensor> recurrent_gla_step_bwd(
    const torch::Tensor& dout,
    const c10::optional<torch::Tensor>& ds_next,
    const torch::Tensor& q_t,
    const torch::Tensor& k_t,
    const torch::Tensor& v_t,
    const torch::Tensor& s_prev,
    const torch::Tensor& alpha_t
);
