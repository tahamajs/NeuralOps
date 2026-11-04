// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor fused_svd_projection_fwd(
    const torch::Tensor& X,
    const torch::Tensor& U,
    const torch::Tensor& S,
    const torch::Tensor& V_T,
    const torch::Tensor& X_res,
    const torch::Tensor& bias,
    double alpha,
    double beta
);

torch::Tensor fused_svd_projection_bwd_dx(
    const torch::Tensor& dOut,
    const torch::Tensor& U,
    const torch::Tensor& S,
    const torch::Tensor& V_T,
    double alpha
);
