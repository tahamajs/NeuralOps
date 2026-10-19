// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor fp8_gemm_fwd(
    const torch::Tensor& A,
    const torch::Tensor& B,
    const torch::Tensor& scale_a,
    const torch::Tensor& scale_b,
    const c10::optional<torch::Tensor>& bias,
    const torch::ScalarType& out_dtype
);
