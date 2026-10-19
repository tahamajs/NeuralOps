// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
torch::Tensor rwkv6_time_mix_fwd(const torch::Tensor& r, const torch::Tensor& k, const torch::Tensor& v, const torch::Tensor& w, const torch::Tensor& u);
