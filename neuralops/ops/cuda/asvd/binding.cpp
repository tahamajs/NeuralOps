// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor asvd_fwd(
    const torch::Tensor& U,
    const torch::Tensor& V,
    const c10::optional<torch::Tensor>& row_gain
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("asvd_fwd", &asvd_fwd, "Activation-Aware SVD Forward (CUDA)");
}
