// Copyright (c) 2024-2026, Taha Majlesi
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// For a list of all contributors, visit:
//   https://github.com/tahamajs/NeuralOps/graphs/contributors

#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor rpca_sparse_svd_fwd(
    const torch::Tensor& x,
    const torch::Tensor& v_r,
    const torch::Tensor& u_r,
    const c10::optional<torch::Tensor>& s,
    const c10::optional<torch::Tensor>& bias
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def(
        "rpca_sparse_svd_fwd",
        &rpca_sparse_svd_fwd,
        "RPCA Sparse SVD forward (CUDA)",
        py::arg("x"),
        py::arg("v_r"),
        py::arg("u_r"),
        py::arg("s") = py::none(),
        py::arg("bias") = py::none()
    );
    m.def("forward", &rpca_sparse_svd_fwd, "RPCA Sparse SVD forward (CUDA)");
}
