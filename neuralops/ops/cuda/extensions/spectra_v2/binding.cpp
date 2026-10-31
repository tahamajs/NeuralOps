#include <torch/extension.h>

torch::Tensor spectra_v2_forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    double scale
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &spectra_v2_forward_cuda, "spectra_v2 forward (CUDA)");
}
