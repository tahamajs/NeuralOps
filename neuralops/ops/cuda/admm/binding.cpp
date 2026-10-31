// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor admm_2of4_forward(
    const torch::Tensor& W
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("admm_2of4_forward", &admm_2of4_forward, "ADMM 2:4 Structured Pruning Forward (CUDA)");
}
