// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor token_compressed_attention_forward_cuda(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    int64_t num_projected_tokens,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "token_compressed_attention_forward",
        &token_compressed_attention_forward_cuda,
        "Token-Compressed Attention forward (CUDA)",
        py::arg("q"),
        py::arg("k"),
        py::arg("v"),
        py::arg("num_projected_tokens"),
        py::arg("scale")
    );
}
