// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor fwsvd_fwd(
    const torch::Tensor& U,
    const torch::Tensor& V
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fwsvd_fwd", &fwsvd_fwd, "Fisher-Weighted SVD Forward (CUDA)");
}
