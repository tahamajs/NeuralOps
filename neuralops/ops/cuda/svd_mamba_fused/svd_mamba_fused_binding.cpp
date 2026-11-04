// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor svd_mamba_fused_fwd(
    const torch::Tensor& u,
    const torch::Tensor& delta,
    const torch::Tensor& A,
    const torch::Tensor& B_ssm,
    const torch::Tensor& C_ssm,
    const c10::optional<torch::Tensor>& D_param,
    const c10::optional<torch::Tensor>& z,
    const c10::optional<torch::Tensor>& global_mix
);
