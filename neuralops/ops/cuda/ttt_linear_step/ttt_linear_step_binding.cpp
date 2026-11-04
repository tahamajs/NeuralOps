// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor ttt_linear_step_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& etas
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "ttt_linear_step_forward",
        &ttt_linear_step_forward_cuda,
        "TTT Linear Attention Step Forward (CUDA C++)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("etas")
    );
}
