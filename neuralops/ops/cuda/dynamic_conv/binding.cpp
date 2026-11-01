// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor dynamic_conv_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& v,
    const torch::Tensor& weights,
    int64_t kernel_size
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "dynamic_conv_attention_forward",
        &dynamic_conv_attention_forward_cuda,
        "Dynamic Depthwise Convolution Attention forward (CUDA)",
        py::arg("q"),
        py::arg("v"),
        py::arg("weights"),
        py::arg("kernel_size")
    );
}
