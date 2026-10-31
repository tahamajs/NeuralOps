// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor hnc_svd_fwd(
    const torch::Tensor& A,
    const torch::Tensor& B
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hnc_svd_fwd", &hnc_svd_fwd, "HNC-SVD Forward (CUDA)");
}
