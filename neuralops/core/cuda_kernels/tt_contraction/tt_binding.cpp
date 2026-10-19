// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor tt_contraction_fwd(
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1
);

torch::Tensor tt_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& core0,
    const torch::Tensor& core1,
    const int p0,
    const int p1,
    const int q0,
    const int q1,
    const c10::optional<torch::Tensor>& bias
);
