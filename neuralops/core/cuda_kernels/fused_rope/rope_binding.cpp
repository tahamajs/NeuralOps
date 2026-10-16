#include <torch/extension.h>

torch::Tensor fused_rope_fwd(
    const torch::Tensor& x,
    const torch::Tensor& cos,
    const torch::Tensor& sin
);

torch::Tensor fused_rope_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& cos,
    const torch::Tensor& sin
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_rope_fwd", &fused_rope_fwd, "Fused RoPE Forward (CUDA)");
    m.def("fused_rope_bwd", &fused_rope_bwd, "Fused RoPE Backward (CUDA)");
}
