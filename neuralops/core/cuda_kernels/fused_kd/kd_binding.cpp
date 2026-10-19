// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor fused_kd_loss_fwd(
    const torch::Tensor& student_logits,
    const torch::Tensor& teacher_logits,
    const float temperature,
    const std::string& reduction
);
