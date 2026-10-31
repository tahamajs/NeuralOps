// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor retnet_chunk_retention_fwd(
    const torch::Tensor& q, const torch::Tensor& k, const torch::Tensor& v, const float gamma
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("retnet_chunk_retention_fwd", &retnet_chunk_retention_fwd, "RetNet Chunk Retention Forward (CUDA)");
}
