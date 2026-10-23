// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor spectral_entropy_svd_gate_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const int effective_rank,
    const c10::optional<torch::Tensor>& s_weights,
    const c10::optional<torch::Tensor>& bias
);
