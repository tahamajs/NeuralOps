#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor dynamic_conv1d_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const c10::optional<torch::Tensor>& bias_opt = c10::nullopt
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dynamic_conv1d_fwd", &dynamic_conv1d_fwd, "dynamic_conv1d forward (CUDA)",
          py::arg("x"), py::arg("weight"), py::arg("bias_opt") = py::none());
}

