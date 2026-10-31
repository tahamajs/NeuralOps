// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor gated_deltanet_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& beta, const c10::optional<torch::Tensor>& g
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gated_deltanet_fwd", &gated_deltanet_fwd, "Gated DeltaNet Forward (CUDA)");
}
