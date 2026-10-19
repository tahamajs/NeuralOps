// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor fused_ce_fwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const int64_t ignore_index
);

torch::Tensor fused_ce_bwd(
    const torch::Tensor& logits,
    const torch::Tensor& targets,
    const torch::Tensor& grad_loss,
    const int64_t ignore_index
);
