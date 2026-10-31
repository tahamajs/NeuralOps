// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor rwkv6_state_scan_fwd(
    const torch::Tensor& r, const torch::Tensor& k, const torch::Tensor& v,
    const torch::Tensor& w, const torch::Tensor& u
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rwkv6_state_scan_fwd", &rwkv6_state_scan_fwd, "RWKV-6 State Scan Forward (CUDA)");
}
