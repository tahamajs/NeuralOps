#include <torch/extension.h>
#include <c10/util/Optional.h>

torch::Tensor hgrn2_spatial_scan_fwd(const torch::Tensor& x, const torch::Tensor& g);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("hgrn2_spatial_scan_fwd", &hgrn2_spatial_scan_fwd, "hgrn2 spatial scan forward (CUDA)");
}

