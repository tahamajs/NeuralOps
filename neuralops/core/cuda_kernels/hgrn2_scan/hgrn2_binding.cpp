#include <torch/extension.h>

torch::Tensor hgrn2_bidirectional_scan_fwd(
    const torch::Tensor& x,
    const torch::Tensor& g
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgrn2_bidirectional_scan_fwd", &hgrn2_bidirectional_scan_fwd, "HGRN2 Bidirectional Scan Forward (CUDA)");
}
