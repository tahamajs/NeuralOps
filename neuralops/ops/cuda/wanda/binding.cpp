// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor wanda_prune_forward(
    const torch::Tensor& W,
    double sparsity,
    const c10::optional<torch::Tensor>& act_norm_opt
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("wanda_prune_forward", &wanda_prune_forward, "Wanda Prune Forward (CUDA)");
}
