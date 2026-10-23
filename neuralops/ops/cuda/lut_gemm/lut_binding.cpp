// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor lut_codebook_dequant_gemm_fwd(
    const torch::Tensor& x,
    const torch::Tensor& indices_packed,
    const torch::Tensor& codebook,
    const int group_size,
    const torch::Tensor& bias
);
