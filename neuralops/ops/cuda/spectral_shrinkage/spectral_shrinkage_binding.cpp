// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <vector>

std::vector<torch::Tensor> spectral_shrinkage_fwd(const torch::Tensor& x_re, const torch::Tensor& x_im, const torch::Tensor& w_re, const torch::Tensor& w_im, const torch::Tensor& tau);
