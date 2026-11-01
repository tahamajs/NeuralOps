// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor cross_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "cross_attention_forward",
        &cross_attention_forward_cuda,
        "Cross Attention forward (CUDA)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("scale")
    );
}
