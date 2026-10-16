#include <torch/extension.h>
#include <vector>

std::vector<torch::Tensor> fused_norm_linear_fwd(
    const torch::Tensor& x,
    const torch::Tensor& weight,
    const torch::Tensor& bias,
    const float eps
);

torch::Tensor fused_norm_linear_bwd_dx(
    const torch::Tensor& dout,
    const torch::Tensor& weight,
    const torch::Tensor& x,
    const float eps
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_norm_linear_fwd", &fused_norm_linear_fwd, "Fused Norm Linear Forward (CUDA)");
    m.def("fused_norm_linear_bwd_dx", &fused_norm_linear_bwd_dx, "Fused Norm Linear Backward dX (CUDA)");
}
