// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>

torch::Tensor direct_svd_fwd(
    const torch::Tensor& U,
    const torch::Tensor& V,
    float scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("direct_svd_fwd", &direct_svd_fwd, "Direct SVD Forward Reconstruction (CUDA)");
}
