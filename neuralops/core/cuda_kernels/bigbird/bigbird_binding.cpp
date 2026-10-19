// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
torch::Tensor bigbird_fwd(const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, int window_size, int num_global_tokens, int num_random_blocks);
