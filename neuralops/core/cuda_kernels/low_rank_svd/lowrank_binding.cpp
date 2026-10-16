#include <torch/extension.h>

torch::Tensor fused_lowrank_gated_gemm_fwd(
    const torch::Tensor& X,
    const torch::Tensor& V,
    const torch::Tensor& U,
    const torch::Tensor& alpha,
    const torch::Tensor& bias
);

torch::Tensor fused_lowrank_gated_gemm_bwd(
    const torch::Tensor& dout,
    const torch::Tensor& X,
    const torch::Tensor& V,
    const torch::Tensor& U,
    const torch::Tensor& alpha
);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_lowrank_gated_gemm_fwd", &fused_lowrank_gated_gemm_fwd, "Fused Low-Rank Gated GEMM Forward (CUDA)");
    m.def("fused_lowrank_gated_gemm_bwd", &fused_lowrank_gated_gemm_bwd, "Fused Low-Rank Gated GEMM Backward (CUDA)");
}
