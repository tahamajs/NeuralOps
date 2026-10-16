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
