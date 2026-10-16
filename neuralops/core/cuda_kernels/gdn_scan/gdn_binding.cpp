#include <torch/extension.h>

torch::Tensor chunked_delta_gate_scan_fwd(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& alpha,
    const torch::Tensor& beta
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("chunked_delta_gate_scan_fwd", &chunked_delta_gate_scan_fwd, "Chunked Delta-Gate Recurrent Scan Forward (CUDA)");
}
