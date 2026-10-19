// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
torch::Tensor samba_m3_mix_fwd(const torch::Tensor& mamba1, const torch::Tensor& mamba2, const torch::Tensor& linear_attn, const torch::Tensor& gate3);
