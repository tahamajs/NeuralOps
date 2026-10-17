#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor logcumsumexp_fwd(const torch::Tensor& x, int dim = -1);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("logcumsumexp_fwd", &logcumsumexp_fwd, "logcumsumexp forward (CUDA)",
          py::arg("x"), py::arg("dim") = -1);
}

